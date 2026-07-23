"""Client Mistral fin : OCR + chat, sélection de modèle, retries.

Les nœuds LangGraph portent la logique métier ; ce module ne fait qu'encapsuler
le SDK `mistralai`, la gestion d'erreurs et les tentatives. Il expose trois
méthodes seulement : `ocr`, `chat_text`, `chat_json`.
"""
from __future__ import annotations

import base64
import json
import random
import time
from typing import Any, Dict, List, Optional

from mistralai.client import Mistral

from .config import MODEL_OCR


class MistralError(RuntimeError):
    """Erreur générique côté Mistral (réseau, quota, réponse malformée)."""


class OCRError(MistralError):
    """Échec spécifique de l'étape OCR."""


# Exceptions transitoires que l'on retente : rate-limit, timeout réseau, 5xx, et
# la saturation de capacité du tier gratuit (429 / code 3505 « capacity exceeded »).
_TRANSIENT_MARKERS = (
    "rate limit", "timeout", "timed out", "429", "500", "502", "503", "504",
    "capacity", "service_tier", "3505",
)


def _is_transient(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


class MistralClient:
    """Wrapper synchrone autour du SDK Mistral, avec retries exponentiels."""

    def __init__(self, api_key: str, *, max_retries: int = 4, base_delay: float = 2.0):
        self._client = Mistral(api_key=api_key)
        self._max_retries = max_retries
        self._base_delay = base_delay

    # -- retry helper ---------------------------------------------------------
    def _with_retries(self, fn, *, what: str):
        last: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001 - on requalifie ci-dessous
                last = exc
                if not _is_transient(exc) or attempt == self._max_retries - 1:
                    break
                # Backoff exponentiel plafonné + jitter (évite les rafales
                # synchronisées quand le tier gratuit est momentanément saturé).
                delay = min(self._base_delay * (2 ** attempt), 20.0) + random.uniform(0, 0.6)
                time.sleep(delay)
        raise MistralError(f"Échec {what} après {self._max_retries} tentative(s) : {last}") from last

    # -- OCR ------------------------------------------------------------------
    def ocr(self, data: bytes, mime: str) -> str:
        """Envoie un document (PDF ou image) à Mistral OCR et renvoie le texte.

        >>> ZONE DÉLICATE #1 — appel Mistral OCR <<<
        Le SDK attend une *référence* de document, pas des octets bruts. On
        encode donc le fichier en Data-URI base64 et on choisit le bon type de
        conteneur :
          - PDF   -> {"type": "document_url", "document_url": "data:application/pdf;base64,..."}
          - image -> {"type": "image_url",    "image_url":    "data:image/png;base64,..."}
        La réponse contient une liste de `pages`, chacune exposant du markdown ;
        on les concatène pour obtenir un texte unique tenant dans le contexte du
        LLM (aucun RAG, cf. cahier des charges).
        """
        b64 = base64.b64encode(data).decode("ascii")
        is_pdf = "pdf" in (mime or "").lower()
        if is_pdf:
            document: Dict[str, Any] = {
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{b64}",
            }
        else:
            # défaut image ; on force un mime image cohérent
            img_mime = mime if (mime or "").startswith("image/") else "image/png"
            document = {
                "type": "image_url",
                "image_url": f"data:{img_mime};base64,{b64}",
            }

        def _call():
            resp = self._client.ocr.process(model=MODEL_OCR, document=document)
            pages = getattr(resp, "pages", None) or []
            texts = [getattr(p, "markdown", None) or getattr(p, "text", "") or "" for p in pages]
            joined = "\n\n".join(t for t in texts if t).strip()
            if not joined:
                raise OCRError("OCR : aucun texte exploitable renvoyé par le modèle.")
            return joined

        try:
            return self._with_retries(_call, what="OCR")
        except OCRError:
            raise
        except MistralError as exc:
            raise OCRError(str(exc)) from exc

    # -- Chat ------------------------------------------------------------------
    def chat_text(
        self, model: str, system: str, user: str, *,
        temperature: float = 0.0, fallback_model: Optional[str] = None,
    ) -> str:
        """Complétion texte simple. Renvoie le contenu string du modèle.

        `fallback_model` : si le modèle principal échoue de façon transitoire
        (ex. `small` saturé → 429 capacity), on rejoue une fois sur ce modèle.
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        def _call(m: str):
            resp = self._client.chat.complete(model=m, messages=messages, temperature=temperature)
            return (resp.choices[0].message.content or "").strip()

        return self._call_with_fallback(_call, model, fallback_model, what="chat")

    def chat_json(
        self, model: str, system: str, user: str, *,
        temperature: float = 0.0, fallback_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Complétion contrainte en JSON. Renvoie un dict parsé.

        Utilise `response_format=json_object` pour fiabiliser la sortie, puis
        parse ; une réponse non-JSON lève MistralError (gérée en amont).
        `fallback_model` : voir `chat_text`.
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        def _call(m: str):
            resp = self._client.chat.complete(
                model=m,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            content = (resp.choices[0].message.content or "").strip()
            try:
                return json.loads(content)
            except json.JSONDecodeError as exc:
                raise MistralError(f"Réponse JSON malformée du modèle : {exc}") from exc

        return self._call_with_fallback(_call, model, fallback_model, what="chat_json")

    def _call_with_fallback(self, call, model: str, fallback_model: Optional[str], *, what: str):
        """Tente `model` (avec retries) ; en cas d'échec transitoire, bascule
        une fois sur `fallback_model` s'il est fourni et différent."""
        try:
            return self._with_retries(lambda: call(model), what=f"{what}({model})")
        except MistralError:
            if fallback_model and fallback_model != model:
                return self._with_retries(
                    lambda: call(fallback_model), what=f"{what}({fallback_model}, secours)"
                )
            raise

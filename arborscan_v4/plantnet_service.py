from __future__ import annotations

import os
import time
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import cv2
import requests

from config import settings


class PlantNetClient:
    def __init__(self) -> None:
        self.api_key = settings.plantnet_api_key
        self.base_url = os.getenv(
            "PLANTNET_BASE_URL",
            "https://my-api.plantnet.org/v2",
        ).rstrip("/")
        self.project = os.getenv("PLANTNET_PROJECT", "all").strip() or "all"
        self.lang = os.getenv("PLANTNET_LANG", "ru").strip() or "ru"
        self.timeout_sec = float(os.getenv("PLANTNET_TIMEOUT_SEC", "20"))
        self.top_k = max(1, min(int(os.getenv("PLANTNET_TOP_K", "3")), 10))
        self.min_score = max(
            0.0,
            min(float(os.getenv("PLANTNET_MIN_SCORE", "0.05")), 1.0),
        )

    @staticmethod
    def _unknown(status: str, message: Optional[str] = None) -> Dict[str, Any]:
        return {
            "status": status,
            "display_name": "Неизвестно",
            "scientific_name": None,
            "common_names": [],
            "confidence": None,
            "top_results": [],
            "predicted_organs": [],
            "remaining_requests": None,
            "latency_ms": None,
            "message": message,
            "source": "plantnet", "engine_version": None,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "score_interpretation": "provider_ranking_score_not_measured_accuracy",
            "taxon_id": None, "taxon_rank": None, "russian_name": None,
        }

    @staticmethod
    def _parse_item(item: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(item, dict):
            item = {}
        species = item.get("species") if isinstance(item, dict) else {}
        if not isinstance(species, dict):
            species = {}

        common_names = species.get("commonNames") or []
        if not isinstance(common_names, list):
            common_names = []
        common_names = [str(v).strip() for v in common_names if str(v).strip()]

        scientific_name = (
            species.get("scientificNameWithoutAuthor")
            or species.get("scientificName")
            or item.get("scientificName")
        )
        scientific_name = str(scientific_name).strip() if scientific_name else None
        display_name = common_names[0] if common_names else (scientific_name or "Неизвестно")

        try:
            confidence = round(float(item.get("score")), 6)
            if not math.isfinite(confidence) or not 0 <= confidence <= 1: confidence = None
        except (TypeError, ValueError):
            confidence = None

        return {
            "display_name": display_name,
            "scientific_name": scientific_name,
            "common_names": common_names[:8],
            "confidence": confidence,
            "taxon_id": str((item.get('gbif') or {}).get('id')) if (item.get('gbif') or {}).get('id') is not None else None,
            "taxon_id_source": "GBIF" if (item.get('gbif') or {}).get('id') is not None else None,
            # Author abbreviations in scientificName do not prove a species.
            "taxon_rank": "species" if isinstance(species.get('scientificNameWithoutAuthor'),str) and len(species['scientificNameWithoutAuthor'].split()) >= 2 else "genus_or_unresolved",
            "russian_name": None,
        }

    def identify(self, crop_bgr) -> Dict[str, Any]:
        if not self.api_key:
            return self._unknown("not_configured", "PLANTNET_API_KEY is not configured")
        if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
            return self._unknown("empty_crop", "Tree crop is empty")

        ok, encoded = cv2.imencode(
            ".jpg",
            crop_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), 92],
        )
        if not ok:
            return self._unknown("encode_error", "Could not encode tree crop")

        endpoint = f"{self.base_url}/identify/{self.project}"
        params = {
            "api-key": self.api_key,
            "lang": self.lang,
            "nb-results": self.top_k,
        }
        files = [("images", ("tree_crop.jpg", encoded.tobytes(), "image/jpeg"))]
        data = [("organs", "auto")]

        started = time.perf_counter()
        try:
            response = requests.post(
                endpoint,
                params=params,
                files=files,
                data=data,
                timeout=self.timeout_sec,
            )
            latency_ms = round((time.perf_counter() - started) * 1000, 1)
            if response.status_code != 200:
                return self._unknown(
                    "http_error",
                    f"Pl@ntNet returned HTTP {response.status_code}",
                ) | {"latency_ms": latency_ms}

            payload = response.json()
            raw_results = payload.get("results") or []
            if not isinstance(raw_results, list) or not raw_results:
                result = self._unknown("no_results", "Pl@ntNet returned no species candidates")
                result.update(
                    {
                        "predicted_organs": payload.get("predictedOrgans") or [],
                        "remaining_requests": payload.get("remainingIdentificationRequests"),
                        "latency_ms": latency_ms,
                    }
                )
                return result

            top_results: List[Dict[str, Any]] = [
                self._parse_item(item) for item in raw_results[: self.top_k]
            ]
            best = top_results[0]
            confidence = best.get("confidence")
            status = "ok"
            message = None
            if confidence is not None and confidence < self.min_score:
                status = "low_confidence"
                message = (
                    "Best Pl@ntNet score is below threshold "
                    f"({confidence:.4f} < {self.min_score:.4f})"
                )

            if confidence is None or not best.get('scientific_name'):
                status = 'invalid_candidate'
            selected = best if status == 'ok' else {
                'display_name':'Неизвестно','scientific_name':None,'confidence':None,
                'common_names':[],'taxon_id':None,'taxon_rank':None,'russian_name':None}

            return {
                "status": status,
                **selected,
                "top_results": top_results,
                "predicted_organs": payload.get("predictedOrgans") or [],
                "remaining_requests": payload.get("remainingIdentificationRequests"),
                "latency_ms": latency_ms,
                "message": message,
                "source":"plantnet", "engine_version":str(payload['version']) if payload.get('version') is not None else None,
                "retrieved_at":datetime.now(timezone.utc).isoformat(),
                "score_interpretation":"provider_ranking_score_not_measured_accuracy",
                # Do not retain provider query URLs (may contain API credentials).
                "original_prediction": {key:payload.get(key) for key in ('results','otherResults','bestMatch','version')},
            }
        except requests.Timeout:
            return self._unknown("timeout", "Pl@ntNet request timed out")
        except requests.RequestException:
            return self._unknown("network_error", "Pl@ntNet connection failed")
        except ValueError:
            return self._unknown("invalid_response", "Pl@ntNet returned invalid JSON")
        except Exception:  # defensive: classification must not crash analysis
            return self._unknown("internal_error", "Invalid identification response")

    def health(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "configured": bool(self.api_key),
            "reachable": None,
            "status": None,
            "status_code": None,
            "latency_ms": None,
            "error": None,
        }
        if not self.api_key:
            return info

        started = time.perf_counter()
        try:
            response = requests.get(
                f"{self.base_url}/_status",
                params={"api-key": self.api_key},
                timeout=min(self.timeout_sec, 10.0),
            )
            info["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
            info["status_code"] = response.status_code
            if response.status_code == 200:
                payload = response.json()
                info["status"] = payload.get("status")
                info["reachable"] = payload.get("status") == "ok"
            else:
                info["reachable"] = False
                info["error"] = f"HTTP {response.status_code}"
        except Exception:
            info["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
            info["reachable"] = False
            info["error"] = "Pl@ntNet health request failed"
        return info

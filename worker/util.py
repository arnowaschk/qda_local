# All comments in English.
import re
import logging
from typing import List, Dict

# Configure logging
logger = logging.getLogger(__name__)

TECH_KEYWORDS = {
    "AI": ["KI", "AI", "künstliche Intelligenz", "Edge", "Computer Vision", "Modell"],
    "Sensors": ["Sensor", "Thermal", "akustisch", "Gas", "Mehrkriterien", "Kamera"],
    "Robotics": ["Roboter", "Drohne", "Kettenroboter", "autonom"],
    "Materials": ["intumeszierend", "Beschichtung", "Aerogel", "Festkörperzelle", "Holz", "Kapselung"],
    "Batteries": ["Batterie", "Akkus", "Speicher", "Thermal Runaway", "USV"],
    "Timber": ["Holz", "Hybrid", "CLT", "sichtbar", "Kapselung"],
    "Norms": ["Norm", "Bauordnung", "Zulassung", "AbP", "CEN", "EN"],
    "LifeSafety": ["Evakuierung", "Flucht", "Räumung", "Pflegeheim", "Krankenhaus"],
    "Sustainability": ["CO2", "Nachhalt", "Energie", "Gründach", "PV", "Dämmung"],
}

STANCE_AXES = {
    "Automation_AI": {
        "pro": ["KI", "autonom", "Edge", "Vorhersage", "Predictive", "Digital Twin"],
        "contra": ["Gadget", "überbewertet", "Fehlalarme", "vertraue", "Verantwortlichkeiten"],
    },
    "Timber_Risk": {
        "pro": ["Kapselung", "Nachweis", "Schichtaufbau", "kontrolliert"],
        "contra": ["Brandlast", "Hohlraum", "Brandautobahn", "kritisch"],
    },
    "Battery_Strictness": {
        "pro": ["Gasableitung", "Inertisierung", "Brandabschnitt", "Frühdetektion"],
        "contra": ["überzogen", "Kosten", "Betrieb", "Hürden"],
    },
    "Sustainability_vs_Safety": {
        "pro": ["Prüfung", "Sicher", "Grenzen", "Safety-Overlay"],
        "contra": ["Freibrief", "unterlaufen", "Risiko"],
    }
}

def keyword_hits(text: str, dictlist: Dict[str, List[str]]) -> Dict[str,int]:
    logger.debug(f"Analyzing keyword hits for text of length {len(text)}")
    t = text.lower()
    out = {}
    total_hits = 0
    for k, kws in dictlist.items():
        hits = sum(1 for w in kws if re.search(r"\\b"+re.escape(w.lower())+r"\\b", t))
        out[k] = hits
        total_hits += hits
        if hits > 0:
            logger.debug(f"Category '{k}': {hits} hits")
    
    if total_hits > 0:
        logger.debug(f"Total keyword hits: {total_hits}")
    return out

def stance(text: str) -> dict:
    logger.debug(f"Computing stance analysis for text of length {len(text)}")
    res = {}
    for axis, buckets in STANCE_AXES.items():
        pro = sum(keyword_hits(text, {"p": buckets["pro"]})["p"] for _ in [0])
        con = sum(keyword_hits(text, {"c": buckets["contra"]})["c"] for _ in [0])
        score = (pro - con)
        res[axis] = score
        if pro > 0 or con > 0:
            logger.debug(f"Stance {axis}: pro={pro}, contra={con}, score={score}")
    return res

def apply_policies(text: str, policies: dict) -> List[str]:
    logger.debug(f"Applying policies to text of length {len(text)}")
    codes = []
    t = text
    
    if not policies.get("codes"):
        logger.debug("No policy codes defined, returning empty list")
        return codes
    
    for item in policies.get("codes", []):
        name = item.get("name")
        any_list = item.get("any", [])
        hit = False
        
        for pat in any_list:
            try:
                if re.search(pat, t, flags=re.IGNORECASE):
                    hit = True
                    logger.debug(f"Policy pattern '{pat}' matched for code '{name}'")
                    break
            except re.error:
                # Fallback to word boundary search if regex is invalid
                if re.search(r"\\b"+re.escape(pat)+r"\\b", t, flags=re.IGNORECASE):
                    hit = True
                    logger.debug(f"Policy pattern '{pat}' matched (word boundary) for code '{name}'")
                    break
        
        if hit and name:
            codes.append(name)
    
    if codes:
        logger.debug(f"Applied policies resulted in codes: {codes}")
    else:
        logger.debug("No policy codes matched")
    
    return sorted(set(codes))

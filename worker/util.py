# All comments in English.
import datetime
import chardet
import sys
import re
import logging
import signal
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

def keyword_hits(text: str, dictlist: Dict[str, List[str]]) -> Dict[str, int]:
    """Count keyword hits in text against a dictionary of keyword lists.
    
    Args:
        text: Input text to search for keywords
        dictlist: Dictionary mapping categories to lists of keywords
        
    Returns:
        Dictionary mapping categories to hit counts
    """
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
    """Analyze text for stance across predefined axes.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary mapping stance axes to numerical scores
    """
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
    """Apply coding policies to text and return matching policy codes.
    
    Args:
        text: Text to analyze against policies
        policies: Dictionary containing policy definitions
        
    Returns:
        List of policy codes that match the text
    """
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

class TimeoutError(Exception):
    """Custom timeout exception"""
    pass

def timeout_handler(signum, frame):
    """Handle timeout signal by raising TimeoutError.
    
    Args:
        signum: Signal number
        frame: Current stack frame
        
    Raises:
        TimeoutError: When the timeout is reached
    """
    from pipeline import TIMEOUT_SECONDS
    raise TimeoutError(f"Pipeline timed out after {TIMEOUT_SECONDS} seconds ({TIMEOUT_SECONDS/3600:.1f} hours)")

def setup_timeout():
    """Setup timeout handler for the current process.
    
    Configures a signal handler that will raise TimeoutError after
    the global TIMEOUT_SECONDS period.
    """
    from pipeline import TIMEOUT_SECONDS
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(TIMEOUT_SECONDS)
        logger.info(f"Timeout set to {TIMEOUT_SECONDS} seconds ({TIMEOUT_SECONDS/3600:.1f} hours)")
    except Exception as e:
        logger.warning(f"Could not set timeout signal: {e}")

def clear_timeout():
    """Clear any pending timeout signal.
    
    This prevents the timeout from being triggered if called before the timeout
    period elapses.
    """
    try:
        signal.alarm(0)
        logger.info("Timeout cleared")
    except Exception as e:
        logger.warning(f"Could not clear timeout: {e}")

class DirectLogger:
    """A logger that writes directly to stderr with enhanced formatting.
    
    This logger adds timestamps, caller information, and consistent formatting
    to all log messages.
    
    Args:
        name: Name to identify the logger in log messages
    """
    def __init__(self, name):
        self.name = name
    
    def _get_caller_info(self) -> str:
        """Get the filename and line number of the code that called the logger.
        
        Returns:
            String in format 'filename:linenumber' or 'unknown:0' if not found
        """
        import inspect
        frame = inspect.currentframe()
        try:
            # Go back 3 frames to get the actual caller:
            # 1. _log
            # 2. debug/info/warning/error method
            # 3. The actual caller
            for _ in range(3):
                frame = frame.f_back
                if frame is None:
                    break
            
            if frame:
                return f"{frame.f_code.co_filename}:{frame.f_lineno}"
            return "unknown:0"
        finally:
            del frame
    
    def _log(self, level: str, msg: str, *args):
        """Internal method to handle the actual logging.
        
        Args:
            level: Log level (e.g., 'INFO', 'ERROR')
            msg: Log message, possibly with format specifiers
            *args: Arguments to format into the message
        """
        if args:
            msg = msg % args
        caller = self._get_caller_info()
        now = datetime.datetime.now().strftime("%Y%m%d %H:%M:%S") 
        print(f"{now}|{level} - {self.name} - [{caller}] - {msg}", 
              file=sys.stderr, flush=True)
    
    def debug(self, msg, *args):
        self._log("DEBUG", msg, *args)
    
    def info(self, msg, *args):
        self._log("INFO ", msg, *args)
    
    def warning(self, msg, *args):
        self._log("WARN ", msg, *args)
    
    def error(self, msg, *args):
        self._log("ERROR", msg, *args)
    
    def exception(self, msg, *args, exc_info=None):
        self._log("EXCEPT", msg, *args)
        if exc_info is None:
            exc_info = sys.exc_info()
        if any(exc_info):  # If there's an exception to log
            import traceback
            tb_lines = traceback.format_exception(*exc_info)
            for line in tb_lines:
                for subline in line.rstrip().split('\n'):
                    if subline.strip():
                        print(f"TRACEBACK - {self.name} - {subline}", 
                              file=sys.stderr, flush=True)
    
    def critical(self, msg, *args):
        self._log("CRIT ", msg, *args)
        sys.exit(1)

def detect_encoding(file_path: str, sample_size: int = 1024) -> str:
    """Detect the most likely encoding of a text file.
    
    Args:
        file_path: Path to the file to analyze
        sample_size: Number of bytes to read for detection (default: 1024)
        
    Returns:
        String representing the detected encoding (e.g., 'utf-8', 'iso-8859-1')
        
    Note:
        Uses chardet library for encoding detection. Falls back to 'utf-8' if
        detection fails or returns None.
    """
    
    with open(file_path, 'rb') as f:
        raw_data = f.read(sample_size)
    
    result = chardet.detect(raw_data)
    return result['encoding'] or 'utf-8'

def clean_text(text: str, lang: str = 'de') -> str:
    """
    Clean and normalize text input with language-specific processing.
    
    Args:
        text: Input text to clean
        lang: Language code ('de' for German, 'en' for English, etc.)
        
    Returns:
        Cleaned and normalized text
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Remove any non-printable characters except newlines and tabs
    import re
    text = re.sub(r'[^\x20-\x7E\n\t\räüößÄÜÖ]', ' ', text)
    
    # Normalize whitespace but preserve paragraph breaks
    text = '\n\n'.join(
        ' '.join(part.split()) 
        for part in text.split('\n\n')
    )
    
    # German-specific text normalization
    if lang.lower() == 'de':
        # Handle common German quotation marks and special characters
        text = text.replace('„', '"')
        text = text.replace('"', '"')
        #text = text.replace('''''', "'")
        
        # Replace common German abbreviations with full forms for better analysis
        abbrev_map = {
            r'\bzzgl\.\s*': 'bezüglich ',
            r'\bbzw\.\s*': 'beziehungsweise ',
            r'\bca\.\s*': 'circa ',
            r'\bz\.B\.\s*': 'zum Beispiel ',
            r'\bu\.a\.\s*': 'unter anderem ',
            r'\betc\.\s*': 'und so weiter ',
            r'\binsb\.\s*': 'insbesondere ',
            r'\bggf\.\s*': 'gegebenenfalls ',
            r'\bz\.T\.\s*': 'zum Teil ',
            r'\bi\.d\.R\.\s*': 'in der Regel ',
            r'\bz\.Z\.\s*': 'zur Zeit ',
            r'\bMrd\.\s*': 'Milliarden ',
            r'\bMio\.\s*': 'Millionen ',
            r'\bS\.\s*': 'Seite ',
            r'\bff\.\s*': 'fortfolgende ',
            r'\bNr\.\s*': 'Nummer ',
            r'\bJh\.\s*': 'Jahrhundert ',
        }
        
        for pattern, replacement in abbrev_map.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Handle common German compound word splits
        text = re.sub(r'(\w+)([A-ZÄÖÜ][a-zäöüß]+)', r'\1 \2', text)
    
    # Additional cleaning steps
    text = re.sub(r'\s+', ' ', text)  # Normalize all whitespace
    text = text.strip()
    
    return text

def create_safe_dirname(header: str, index: int) -> str:
    """Create a safe directory name from column header and index.
    
    Args:
        header: The column header text
        index: The column index (for ordering)
        
    Returns:
        str: Safe directory name in format 'NN_first_30_chars'
    """
    # Take first 30 characters, remove invalid path chars, and clean up
    safe_name = re.sub(r'[^\w\s-]', '', header[:30].strip())
    safe_name = re.sub(r'[\s-]+', '_', safe_name).strip('_')
    return f"{index:02d}_{safe_name}"


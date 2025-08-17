# REFI-QDA .qdpx generator (minimal subset).
import csv, json, pathlib, zipfile, datetime
import logging
from xml.etree.ElementTree import Element, SubElement, tostring
from typing import List, Dict

# Configure logging
logger = logging.getLogger(__name__)

def _iso_now() -> str:
    """Generate current UTC time in ISO 8601 format with 'Z' timezone.
    
    Returns:
        str: Current UTC time in ISO 8601 format (e.g., '2023-01-01T12:00:00Z')
        
    Example:
        >>> _iso_now()
        '2023-01-01T12:00:00Z'
    """
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _ensure_list(x: Any) -> List[str]:
    """Convert various input formats to a list of strings.
    
    Handles multiple input formats:
    - List: Returns as-is
    - JSON string: Parses as JSON if possible
    - Semicolon-separated string: Splits on semicolons
    - Comma-separated string: Splits on commas
    - Single value: Returns as single-item list
    
    Args:
        x: Input value to convert (list, str, or any value)
        
    Returns:
        List[str]: List of non-empty strings
        
    Examples:
        >>> _ensure_list("a;b;c")
        ['a', 'b', 'c']
        >>> _ensure_list('["x", "y"]')
        ['x', 'y']
        >>> _ensure_list("single")
        ['single']
    """
    if isinstance(x, list):
        return x
    s = str(x).strip()
    if not s:
        return []
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return v
    except Exception:
        pass
    if ";" in s:
        return [t.strip() for t in s.split(";") if t.strip()]
    if "," in s:
        return [t.strip() for t in s.split(",") if t.strip()]
    return [s]

def _etree_to_bytes(elem: Element) -> bytes:
    """Convert an ElementTree Element to bytes with XML declaration.
    
    Args:
        elem: XML Element to convert
        
    Returns:
        bytes: UTF-8 encoded XML data with XML declaration
        
    Example:
        >>> root = Element("Test")
        >>> _etree_to_bytes(root)
        b'<?xml version="1.0" encoding="UTF-8"?>\n<Test />'
    """
    xml_decl = b'<?xml version="1.0" encoding="UTF-8"?>\n'
    return xml_decl + tostring(elem, encoding="utf-8")

def build_qdpx(coded_segments_csv: str, out_qdpx: str, project_name: str = "QDA Project") -> str:
    """Generate a REFI-QDA (.qde) compatible ZIP archive from coded segments.
    
    This function creates a minimal but valid REFI-QDA project file that can be
    imported into qualitative data analysis software like MAXQDA, NVivo, or Atlas.ti.
    
    The generated archive contains:
    - Project metadata
    - Codebook with all unique codes
    - Documents (one per row in the input CSV)
    - Code assignments linking documents to codes
    
    Args:
        coded_segments_csv: Path to input CSV file with columns:
                           - codes: Semicolon-separated list of codes
                           - text: Document text
                           - set_id: (optional) Set identifier
                           - question_idx: (optional) Question number/index
        out_qdpx: Output path for the .qdpx file
        project_name: Name for the QDA project
        
    Returns:
        str: Path to the generated .qdpx file
        
    Raises:
        FileNotFoundError: If input CSV doesn't exist
        PermissionError: If output directory is not writable
        
    Example:
        >>> build_qdpx("coded_data.csv", "output.qdpx", "My Research Project")
        'output.qdpx'
    """
    logger.info(f"Starting REFI-QDA export - input: {coded_segments_csv}, output: {out_qdpx}")
    
    # Read input CSV
    logger.info("Reading coded segments CSV file...")
    rows: List[Dict] = []
    try:
        with open(coded_segments_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        logger.info(f"Successfully read {len(rows)} rows from CSV")
    except Exception as e:
        logger.error(f"Failed to read CSV file: {e}")
        raise

    # Extract unique codes
    logger.info("Extracting unique codes from segments...")
    code_set = set()
    for r in rows:
        for c in _ensure_list(r.get("codes","")):
            if c:
                code_set.add(str(c))
    logger.info(f"Found {len(code_set)} unique codes")

    # Build XML structure
    logger.info("Building XML project structure...")
    root = Element("QDAProject", {"Version": "1.0", "Creator": "QDA-Local"})
    
    # Project details
    details = SubElement(root, "ProjectDetails")
    SubElement(details, "Name").text = project_name
    SubElement(details, "Description").text = "Auto-generated REFI-QDA export (minimal)."
    SubElement(details, "CaseSensitive").text = "false"
    SubElement(details, "DateTimeCreated").text = _iso_now()
    logger.info("Project details added to XML")

    # Codebook
    logger.info("Building codebook XML...")
    codebook = SubElement(root, "CodeBook")
    codes_xml = SubElement(codebook, "Codes")
    code_id_map = {}
    for i, cname in enumerate(sorted(code_set), start=1):
        cid = f"c{i}"
        code_id_map[cname] = cid
        c = SubElement(codes_xml, "Code", {"id": cid})
        SubElement(c, "Name").text = cname
        SubElement(c, "IsActive").text = "true"
        SubElement(c, "Description").text = ""
    logger.info(f"Codebook XML created with {len(code_id_map)} codes")

    # Sources/Documents
    logger.info("Building documents XML...")
    sources = SubElement(root, "Sources")
    documents = SubElement(sources, "Documents")
    for i, r in enumerate(rows, start=1):
        did = f"d{i}"
        name = f"Set{r.get('set_id','')}_Q{r.get('question_idx','')}"
        text = str(r.get("text",""))
        d = SubElement(documents, "Document", {"id": did})
        SubElement(d, "Name").text = name
        ts = SubElement(d, "TextSource")
        SubElement(ts, "PlainText").text = text
    logger.info(f"Documents XML created with {len(rows)} documents")

    # Code assignments
    logger.info("Building code assignments XML...")
    cas = SubElement(root, "CodeAssignments")
    n = 0
    for i, r in enumerate(rows, start=1):
        did = f"d{i}"
        text = str(r.get("text",""))
        codes = [c for c in _ensure_list(r.get("codes","")) if c in code_id_map]
        start, end = 0, len(text)
        for cname in codes:
            n += 1
            ca = SubElement(cas, "Coding", {"id": f"a{n}"})
            SubElement(ca, "CodeRef", {"id": code_id_map[cname]})
            SubElement(ca, "DocumentRef", {"id": did})
            SubElement(ca, "Segment", {"StartPosition": str(start), "EndPosition": str(end)})
    logger.info(f"Code assignments XML created with {n} assignments")

    # Create output directory and ZIP file
    logger.info("Creating output ZIP file...")
    out_qdpx_path = pathlib.Path(out_qdpx)
    out_qdpx_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with zipfile.ZipFile(out_qdpx, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("project.qde", _etree_to_bytes(root))
        logger.info(f"REFI-QDA ZIP file created successfully: {out_qdpx_path}")
    except Exception as e:
        logger.error(f"Failed to create ZIP file: {e}")
        raise
    
    logger.info(f"REFI-QDA export completed successfully")
    return str(out_qdpx_path)

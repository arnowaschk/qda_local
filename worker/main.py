from fastapi import FastAPI, Query
from pydantic import BaseModel
import subprocess
import pathlib
import sys
import traceback
import logging
import torch

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

# Suppress noisy INFO logs from fontTools (used by WeasyPrint)
try:
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
    logging.getLogger("fontTools.ttLib").setLevel(logging.WARNING)
except Exception:
    pass

# Log CUDA information
logger.info("\n=== CUDA Information ===")
logger.info(f"PyTorch version: {torch.__version__}")
if torch.cuda.is_available():
    logger.info(f"CUDA available: {torch.version.cuda}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    logger.info("CUDA is NOT available. Using CPU.")
logger.info("======================\n")

app = FastAPI(title="QDA Worker API")

class ProcessResponse(BaseModel):
    """Response model for API operations.
    
    Attributes:
        ok (bool): Indicates whether the operation was successful.
        error_message (str): Contains error details if the operation failed.
    """
    ok: bool
    error_message: str = ""

def get_error_location():
    """Extract file and line information from the current traceback.
    
    Returns:
        str: A string containing the file, line number, and function name where the error occurred,
             or a message indicating the location could not be determined.
    """
    try:
        tb = traceback.extract_tb(sys.exc_info()[2])
        if tb:
            frame = tb[-1]  # Get the last frame
            return f"File: {frame.filename}, Line: {frame.lineno}, Function: {frame.name}"
        return "Location: Unknown"
    except Exception:
        return "Location: Could not determine"

def run_pipeline_realtime(pipeline_command):
    """Execute a pipeline command and capture its output in real-time.
    
    Args:
        pipeline_command (list): List of command-line arguments to execute the pipeline.
        
    Returns:
        ProcessResponse: A response object indicating success or failure, including any error messages
                       and the last 20 lines of output if the command fails.
    """
    try:
        logger.info("Starting pipeline with real-time output...")
        process = subprocess.Popen(
            pipeline_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd="/app"
        )
        
        # Capture and log output in real-time
        output_lines = []
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                output = output.strip()
                logger.info(f"PIPELINE: {output}")
                output_lines.append(output)
        
        return_code = process.poll()
        logger.info(f"Pipeline execution completed with return code: {return_code}")
        
        if return_code == 0:
            return ProcessResponse(ok=True, error_message="")
        else:
            error_msg = f"Pipeline failed with exit code {return_code}"
            if output_lines:
                error_msg += "\nLast 20 lines of output:\n"
                error_msg += "\n".join([f"  {line}" for line in output_lines[-20:]])
            logger.error(error_msg)
            return ProcessResponse(ok=False, error_message=error_msg)
            
    except Exception as e:
        error_msg = f"Failed to run pipeline: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return ProcessResponse(ok=False, error_message=error_msg)

@app.get("/health")
def health():
    """Health check endpoint for the worker API.
    
    Returns:
        dict: A dictionary with a status key indicating the service health.
    """
    return {"status": "ok"}

@app.post("/process", response_model=ProcessResponse)
def process(input_path: str = Query(...), out_dir: str = Query(...), llm_model: str = Query(None)):
    """Process input file through the pipeline and save output to specified directory.
    
    Args:
        input_path (str): Path to the input file to be processed.
        out_dir (str): Directory where the output files should be saved.
        
    Returns:
        ProcessResponse: A response object indicating success or failure of the processing.
                       Includes error details if the operation fails.
    """
    logger.info(f"Processing request - input: {input_path}, output: {out_dir}")
    
    try:
        ip = pathlib.Path(input_path)
        od = pathlib.Path(out_dir)
        
        # Ensure output directory exists
        od.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory ready: {od}")
        
        # Verify pipeline module
        pipeline_module_path = pathlib.Path("/app/pipeline.py")
        if not pipeline_module_path.exists():
            raise FileNotFoundError(f"Pipeline module not found at: {pipeline_module_path}")
        
        # Build and execute pipeline command
        pipeline_command = [
            sys.executable, "-m", "pipeline",
            "--input", str(ip),
            "--out", str(od)
        ]
        if llm_model:
            pipeline_command.extend(["--llm-model", llm_model])
        logger.info(f"Executing: {' '.join(pipeline_command)}")
        
        # Run the pipeline
        return run_pipeline_realtime(pipeline_command)
        
    except subprocess.TimeoutExpired:
        error_msg = f"Pipeline execution timed out after 5 hours. {get_error_location()}"
        logger.error(error_msg)
        return ProcessResponse(ok=False, error_message=error_msg)
        
    except Exception as e:
        error_msg = f"Error: {str(e)}\n{get_error_location()}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return ProcessResponse(ok=False, error_message=error_msg)

if __name__ == "__main__":
    """Main entry point for the worker API server.
    
    Starts a uvicorn server to handle incoming API requests for the QDA worker.
    The server runs on all available network interfaces (0.0.0.0) on port 8001.
    """
    logger.info("Worker API server starting")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

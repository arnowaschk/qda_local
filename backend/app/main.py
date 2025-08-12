from fastapi import FastAPI, Query
from pydantic import BaseModel
import subprocess, pathlib
import os
import httpx
import asyncio
import traceback
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="QDA Local API")

class AnalyzeResponse(BaseModel):
    ok: bool
    out_dir: str
    error_message: str = ""

def get_error_location():
    """Extract file and line information from the current traceback"""
    try:
        tb = traceback.extract_tb(sys.exc_info()[2])
        if tb:
            # Get the last frame (most recent call)
            frame = tb[-1]
            return f"File: {frame.filename}, Line: {frame.lineno}, Function: {frame.name}"
        return "Location: Unknown"
    except:
        return "Location: Could not determine"

@app.get("/health")
def health():
    logger.info("Health check endpoint called")
    return {"status": "ok"}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(input_path: str = Query(...), out_dir: str = Query("./out")):
    logger.info(f"Starting analysis request - input_path: {input_path}, out_dir: {out_dir}")
    
    ip = pathlib.Path(input_path)
    od = pathlib.Path(out_dir)
    
    logger.info(f"Resolved paths - input: {ip}, output: {od}")
    
    # Ensure output directory exists
    try:
        od.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory created/verified: {od}")
    except Exception as e:
        logger.error(f"Failed to create output directory: {e}")
        return AnalyzeResponse(
            ok=False, 
            out_dir=str(od), 
            error_message=f"Failed to create output directory: {str(e)}"
        )
    
    try:
        # Try to communicate with worker via HTTP first (more reliable)
        worker_url = "http://qda_worker:8001"
        logger.info(f"Attempting HTTP communication with worker at: {worker_url}")
        
        try:
            async with httpx.AsyncClient(timeout=18000.0) as client:
                logger.info("Sending HTTP request to worker")
                response = await client.post(
                    f"{worker_url}/process",
                    params={
                        "input_path": str(ip),
                        "out_dir": str(od)
                    }
                )
                
                logger.info(f"Worker HTTP response received - status: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Worker processing successful: {result}")
                    return AnalyzeResponse(
                        ok=result.get("ok", False),
                        out_dir=str(od),
                        error_message=result.get("error_message", "")
                    )
                else:
                    location = get_error_location()
                    error_msg = f"Worker communication failed: HTTP {response.status_code}. {location}"
                    logger.error(f"Worker HTTP error: {error_msg}")
                    return AnalyzeResponse(
                        ok=False,
                        out_dir=str(od),
                        error_message=error_msg
                    )
                    
        except httpx.RequestError as e:
            logger.warning(f"HTTP communication failed, falling back to docker exec: {e}")
            # Fallback to docker exec if HTTP fails
            worker_command = [
                "docker", "exec", "qda_worker", "python", "-m", "pipeline", 
                "--input", str(ip), "--out", str(od)
            ]
            
            logger.info(f"Executing docker exec command: {' '.join(worker_command)}")
            
            result = subprocess.run(
                worker_command,
                capture_output=True,
                text=True,
                timeout=18000
            )
            
            logger.info(f"Docker exec completed - return code: {result.returncode}")
            if result.stderr:
                logger.warning(f"Docker exec stderr: {result.stderr}")
            if result.stdout:
                logger.info(f"Docker exec stdout: {result.stdout}")
            
            if result.returncode == 0:
                logger.info("Pipeline execution successful via docker exec")
                return AnalyzeResponse(ok=True, out_dir=str(od), error_message="")
            else:
                error_msg = f"Pipeline failed with exit code {result.returncode}"
                if result.stderr:
                    error_msg += f". Error: {result.stderr.strip()}"
                if result.stdout:
                    error_msg += f". Output: {result.stdout.strip()}"
                
                logger.error(f"Pipeline execution failed: {error_msg}")
                return AnalyzeResponse(ok=False, out_dir=str(od), error_message=error_msg)
            
    except subprocess.TimeoutExpired:
        location = get_error_location()
        error_msg = f"Pipeline execution timed out after set timeout. {location}"
        logger.error(f"Timeout error: {error_msg}")
        return AnalyzeResponse(
            ok=False, 
            out_dir=str(od), 
            error_message=error_msg
        )
    except Exception as e:
        location = get_error_location()
        error_details = f"Unexpected error: {str(e)}. {location}"
        
        logger.error(f"Unexpected error occurred: {error_details}")
        
        # Add full traceback for debugging
        try:
            tb_str = ''.join(traceback.format_exc())
            error_details += f"\n\nFull traceback:\n{tb_str}"
            logger.error(f"Full traceback: {tb_str}")
        except Exception as e:
            print("traceback failed", e)
            logger.error(f"Failed to get traceback: {e}")
            
        return AnalyzeResponse(
            ok=False, 
            out_dir=str(od), 
            error_message=error_details
        )

from fastapi import FastAPI, Query
from pydantic import BaseModel
import subprocess
import pathlib
import sys
import os
import traceback
import logging
import traceback
import torch
# Check CUDA availability and device info
print("\n=== CUDA Information ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device count: {torch.cuda.device_count()}")
else:
    print("CUDA is NOT available. Using CPU.")
print("======================\n")
# Import our direct logger from pipeline
from pipeline import logger

# Set up basic logging to stderr
import sys
import logging

class DirectLogger:
    def __init__(self, name):
        self.name = name
    
    def _log(self, level, msg, *args):
        if args:
            msg = msg % args
        print(f"{level} - {self.name} - {msg}", file=sys.stderr, flush=True)
    
    def debug(self, msg, *args):
        self._log("DEBUG", msg, *args)
    
    def info(self, msg, *args):
        self._log("INFO ", msg, *args)
    
    def warning(self, msg, *args):
        self._log("WARN ", msg, *args)
    
    def error(self, msg, *args):
        self._log("ERROR", msg, *args)
    
    def exception(self, msg, *args):
        self._log("EXCEPTION", msg, *args)
        traceback.print_exc(file=sys.stderr)
    
    def critical(self, msg, *args):
        self._log("CRIT ", msg, *args)
        sys.exit(1)

# Override the root logger
logging.root = DirectLogger('root')
logger = logging.root

# Add global exception handler
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

app = FastAPI(title="QDA Worker API")

class ProcessResponse(BaseModel):
    ok: bool
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

def run_pipeline_realtime(pipeline_command):
    """Run pipeline with real-time output capture"""
    try:
        process = subprocess.Popen(
            pipeline_command,
            cwd="/app",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        logger.info("Subprocess was just started!")
        
        # Capture output in real-time
        stdout_lines = []
        stderr_lines = []
        
        # Use select for non-blocking I/O
        import select
        import time
        start_time = time.time()
        timeout_seconds = 18000  # 5 hours timeout
        
        # Set pipes to non-blocking mode
        import fcntl
        import os
        
        # Make stdout non-blocking
        fd_stdout = process.stdout.fileno()
        fl_stdout = fcntl.fcntl(fd_stdout, fcntl.F_GETFL)
        fcntl.fcntl(fd_stdout, fcntl.F_SETFL, fl_stdout | os.O_NONBLOCK)
        
        # Make stderr non-blocking
        fd_stderr = process.stderr.fileno()
        fl_stderr = fcntl.fcntl(fd_stderr, fcntl.F_GETFL)
        fcntl.fcntl(fd_stderr, fcntl.F_SETFL, fl_stderr | os.O_NONBLOCK)
        
        # Run the pipeline directly to see the error
        logger.info("Running pipeline with direct output...")
        try:
            # Run with Popen to capture output in real-time
            process = subprocess.Popen(
                pipeline_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combine stdout and stderr
                text=True,
                bufsize=1,
                cwd="/app"
            )
            
            # Read output line by line
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    output = output.strip()
                    logger.info(f"PIPELINE: {output}")
                    stdout_lines.append(output)
            
            return_code = process.poll()
            
            if return_code != 0:
                error_msg = f"Pipeline failed with exit code {return_code}"
                if stdout_lines:
                    error_msg += f"\nLast 20 lines of output:\n"
                    error_msg += "\n".join([f"  {line}" for line in stdout_lines[-20:]])
                logger.error(error_msg)
                return ProcessResponse(ok=False, error_message=error_msg)
                
        except Exception as e:
            error_msg = f"Failed to run pipeline: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return ProcessResponse(ok=False, error_message=error_msg)
        
        return_code = process.returncode
        logger.info(f"Worker pipeline execution completed - return code: {return_code}")
        
        if return_code == 0:
            logger.info("Worker pipeline execution successful")
            return ProcessResponse(ok=True, error_message="")
        else:
            error_msg = f"Pipeline failed with exit code {return_code}"
            if stderr_lines:
                error_msg += f". Errors: {'; '.join(stderr_lines[-5:])}"  # Last 5 error lines
            if stdout_lines:
                error_msg += f". Output: {'; '.join(stdout_lines[-5:])}"  # Last 5 output lines
            
            logger.error(f"Worker pipeline execution failed: {error_msg}")
            time.sleep(5)
            logger.error("Worker pipeline execution failed - returning")
            return ProcessResponse(ok=False, error_message=error_msg)
            
    except Exception as subprocess_error:
        logger.error(f"Worker failed to execute pipeline subprocess: {subprocess_error}")
        
        logger.error(f"Subprocess error traceback: {traceback.format_exc()}")
        raise subprocess_error

@app.get("/health")
def health():
    logger.info("Worker health check endpoint called")
    return {"status": "ok"}

@app.post("/process", response_model=ProcessResponse)
def process(input_path: str = Query(...), out_dir: str = Query(...)):
    logger.info(f"Worker received process request - input_path: {input_path}, out_dir: {out_dir}")
    timeout_secs = 18000  # 5 hours timeout

    try:
        ip = pathlib.Path(input_path)
        od = pathlib.Path(out_dir)
        
        logger.info(f"Worker resolved paths - input: {ip}, output: {od}")
        
        # Ensure output directory exists
        try:
            od.mkdir(parents=True, exist_ok=True)
            logger.info(f"Worker created/verified output directory: {od}")
        except Exception as e:
            logger.error(f"Worker failed to create output directory: {e}")
            return ProcessResponse(
                ok=False,
                error_message=f"Failed to create output directory: {str(e)}"
            )
        
        # Run the pipeline
        pipeline_command = [sys.executable, "-m", "pipeline", "--input", str(ip), "--out", str(od)]
        logger.info(f"Worker executing pipeline command: {' '.join(pipeline_command)}")
        logger.info(f"Worker working directory: {os.getcwd()}")
        logger.info(f"Worker Python executable: {sys.executable}")
        logger.info(f"Worker Python version: {sys.version}")
        
        # Check if pipeline module exists
        pipeline_module_path = pathlib.Path("/app/pipeline.py")
        if pipeline_module_path.exists():
            logger.info(f"Pipeline module found at: {pipeline_module_path}")
        else:
            logger.error(f"Pipeline module not found at: {pipeline_module_path}")
            return ProcessResponse(
                ok=False,
                error_message=f"Pipeline module not found at {pipeline_module_path}"
            )
        
        # Test if pipeline module can be imported
        try:
            logger.info("Testing pipeline module import...")
            import importlib.util
            spec = importlib.util.spec_from_file_location("pipeline", "/app/pipeline.py")
            pipeline_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(pipeline_module)
            logger.info("Pipeline module imported successfully")
        except Exception as import_error:
            logger.error(f"Failed to import pipeline module: {import_error}")
            return ProcessResponse(
                ok=False,
                error_message=f"Pipeline module import failed: {import_error}"
            )
        
        # Run pipeline with real-time output capture
        logger.info("Worker starting our pipeline execution...")
        
        # Try real-time capture first, fallback to simple capture if it fails
        try:
            # Method 1: Real-time capture with non-blocking I/O
            logger.info("Attempting real-time output capture...")
            result = run_pipeline_realtime(pipeline_command)
            return result
        except Exception as realtime_error:
            logger.warning(f"Real-time capture failed: {realtime_error}, falling back to simple capture")
            try:
                # Method 2: Simple capture with timeout
                logger.info("Using simple subprocess capture...")
                result = subprocess.run(
                    pipeline_command,
                    cwd="/app",
                    capture_output=True,
                    text=True,
                    timeout=timeout_secs
                )
                
                logger.info(f"Simple capture completed - return code: {result.returncode}")
                
                if result.stderr:
                    logger.warning(f"Pipeline stderr: {result.stderr}")
                if result.stdout:
                    logger.info(f"Pipeline stdout: {result.stdout}")
                
                if result.returncode == 0:
                    logger.info("Worker pipeline execution successful")
                    return ProcessResponse(ok=True, error_message="")
                else:
                    error_msg = f"Pipeline failed with exit code {result.returncode}"
                    if result.stderr:
                        error_msg += f". Error: {result.stderr.strip()}"
                    if result.stdout:
                        error_msg += f". Output: {result.stdout.strip()}"
                    
                    logger.error(f"Worker pipeline execution failed: {error_msg}")
                    return ProcessResponse(ok=False, error_message=error_msg)
                    
            except subprocess.TimeoutExpired:
                logger.error(f"Simple capture also timed out after {timeout_secs} seconds")
                return ProcessResponse(
                    ok=False,
                    error_message=f"Pipeline execution timed out after {timeout_secs} seconds (simple capture)"
                )
            except Exception as simple_error:
                logger.error(f"Simple capture also failed: {simple_error}")
                return ProcessResponse(
                    ok=False,
                    error_message=f"Both capture methods failed: realtime={realtime_error}, simple={simple_error}"
                )
            
        except Exception as subprocess_error:
            logger.error(f"Worker failed to execute pipeline subprocess: {subprocess_error}")
            import traceback
            logger.error(f"Subprocess error traceback: {traceback.format_exc()}")
            return ProcessResponse(
                ok=False,
                error_message=f"Failed to execute pipeline: {str(subprocess_error)}"
            )
            
    except subprocess.TimeoutExpired:
        location = get_error_location()
        error_msg = f"Pipeline execution timed out after {timeout_secs} seconds. {location}"
        logger.error(f"Worker timeout error: {error_msg}")
        return ProcessResponse(
            ok=False,
            error_message=error_msg
        )
    except Exception as e:
        location = get_error_location()
        error_details = f"Unexpected error: {str(e)}. {location}"
        
        logger.error(f"Worker unexpected error: {error_details}")
        
        # Add full traceback for debugging
        try:
            tb_str = ''.join(traceback.format_exc())
            error_details += f"\n\nFull traceback:\n{tb_str}"
            logger.error(f"Worker full traceback: {tb_str}")
        except Exception as traceback_error:
            logger.error(f"Worker failed to get traceback: {traceback_error}")
            
        return ProcessResponse(
            ok=False,
            error_message=error_details
        )

if __name__ == "__main__":
    logger.info("Worker API server starting up")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 
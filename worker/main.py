from fastapi import FastAPI, Query
from pydantic import BaseModel
import subprocess
import pathlib
import sys
import os
import traceback
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        
        logger.info("Pipes set to non-blocking mode")
        
        while True:
            # Check timeout
            if time.time() - start_time > timeout_seconds:
                logger.error(f"Worker pipeline execution timed out after {timeout_seconds} seconds")
                process.terminate()
                try:
                    process.wait(timeout=10)  # Give it 10 seconds to terminate gracefully
                except subprocess.TimeoutExpired:
                    process.kill()  # Force kill if it doesn't terminate
                return ProcessResponse(
                    ok=False,
                    error_message=f"Pipeline execution timed out after {timeout_seconds} seconds"
                )
            
            # Check if process has finished
            if process.poll() is not None:
                logger.info(f"Subprocess finished with return code: {process.returncode}")
                break
            
            # Try to read from pipes with select (non-blocking)
            try:
                ready_to_read, _, _ = select.select([process.stdout, process.stderr], [], [], 0.1)
                
                for pipe in ready_to_read:
                    try:
                        if pipe == process.stdout:
                            line = pipe.readline()
                            if line:
                                line = line.strip()
                                if line:
                                    logger.info(f"Pipeline stdout: {line}")
                                    stdout_lines.append(line)
                        elif pipe == process.stderr:
                            line = pipe.readline()
                            if line:
                                line = line.strip()
                                if line:
                                    logger.warning(f"Pipeline stderr: {line}")
                                    stderr_lines.append(line)
                    except (OSError, IOError) as e:
                        # Pipe might be closed or have no data
                        logger.debug(f"Pipe read error (normal): {e}")
                        pass
                        
            except Exception as select_error:
                logger.debug(f"Select error (normal): {select_error}")
                pass
            
            # Small sleep to prevent busy waiting
            time.sleep(0.1)
        
        # Read any remaining output
        try:
            remaining_stdout, remaining_stderr = process.communicate(timeout=10)
            if remaining_stdout:
                for line in remaining_stdout.strip().split('\n'):
                    if line.strip():
                        logger.info(f"Pipeline stdout (remaining): {line.strip()}")
                        stdout_lines.append(line.strip())
            if remaining_stderr:
                for line in remaining_stderr.strip().split('\n'):
                    if line.strip():
                        logger.warning(f"Pipeline stderr (remaining): {line.strip()}")
                        stderr_lines.append(line.strip())
        except subprocess.TimeoutExpired:
            logger.warning("Timeout reading remaining output, terminating subprocess")
            process.terminate()
            process.wait(timeout=5)
        
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
            return ProcessResponse(ok=False, error_message=error_msg)
            
    except Exception as subprocess_error:
        logger.error(f"Worker failed to execute pipeline subprocess: {subprocess_error}")
        import traceback
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
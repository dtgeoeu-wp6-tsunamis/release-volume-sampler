import logging
import os

def setup_logger(routine_name, log_folder):
    """
    Set up a logger for a specific routine.
    
    Args:
        routine_name (str): The name of the routine (used in the logfile name).
        log_folder (str): The folder where the logfile will be saved.

    Returns:
        logger: Configured logger object.
    """
    # Ensure the log folder exists
    os.makedirs(log_folder, exist_ok=True)
    
    # Define the log file path
    log_file = os.path.join(log_folder, f"{routine_name}.log")
    
    # Create a logger
    logger = logging.getLogger(routine_name)
    logger.setLevel(logging.INFO)  # Set the log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    # Prevent duplication of handlers if logger is already configured
    if not logger.handlers:
        # Create a file handler
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.INFO)
        
        # Create a console (stream) handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create a formatter and set it for both handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger
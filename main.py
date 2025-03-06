import os
import logging
from src.utils import ensure_log_directory

# Set up logging for the main application
log_dir = ensure_log_directory()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "app.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PhysicsApp") 
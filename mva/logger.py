import logging
import os

log = logging.getLogger('analysis')
if not os.environ.get("DEBUG", False):
    log.setLevel(logging.INFO)

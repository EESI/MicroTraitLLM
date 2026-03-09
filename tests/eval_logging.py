import os
from datetime import datetime
from pathlib import Path

class ScriptLogger:
    def __init__(self, log_file='script.log', log_dir=None, append=True):
        """
        Initialize the logger.
        
        Args:
            log_file: Name of the log file (default: 'script.log')
            log_dir: Directory for log file (default: current directory)
            append: If True, append to existing log; if False, overwrite
        """
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self.log_path = os.path.join(log_dir, log_file)
        else:
            self.log_path = log_file
        
        self.mode = 'a' if append else 'w'
        
        # Write initial separator when starting
        self._write_to_file(f"\n{'='*60}")
        self._write_to_file(f"Script started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._write_to_file(f"{'='*60}\n")
    
    def _write_to_file(self, message):
        """Write message to log file."""
        with open(self.log_path, 'a') as f:
            f.write(message + '\n')
    
    def _format_message(self, level, message):
        """Format log message with timestamp and level."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return f"[{timestamp}] [{level}] {message}"
    
    def info(self, message):
        """Log informational message (progress, status updates)."""
        formatted = self._format_message('INFO', message)
        print(formatted)  # Also print to console
        self._write_to_file(formatted)
    
    def error(self, message):
        """Log error message."""
        formatted = self._format_message('ERROR', message)
        print(formatted)
        self._write_to_file(formatted)
    
    def warning(self, message):
        """Log warning message."""
        formatted = self._format_message('WARNING', message)
        print(formatted)
        self._write_to_file(formatted)
    
    def debug(self, message):
        """Log debug message."""
        formatted = self._format_message('DEBUG', message)
        print(formatted)
        self._write_to_file(formatted)
    
    def success(self, message):
        """Log success message."""
        formatted = self._format_message('SUCCESS', message)
        print(formatted)
        self._write_to_file(formatted)
    
    def section(self, message):
        """Log a section header for better organization."""
        separator = '-' * 60
        self._write_to_file(separator)
        formatted = self._format_message('SECTION', message)
        print(formatted)
        self._write_to_file(formatted)
        self._write_to_file(separator)


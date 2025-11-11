"""
Global database configuration settings.
This file contains the centralized database configuration that should be used across the codebase.
"""

# Default settings
DEFAULT_HOST = 'bird_interact_postgresql'  # Default container hostname
DEFAULT_PORT = 5432
DEFAULT_USER = 'root'
DEFAULT_PASSWORD = '123123'
DEFAULT_MINCONN = 1
DEFAULT_MAXCONN = 5

# Global configuration override
GLOBAL_CONFIG = {
    'host': DEFAULT_HOST,  # Default to container hostname
    'port': DEFAULT_PORT,
    'user': DEFAULT_USER,
    'password': DEFAULT_PASSWORD,
    'minconn': DEFAULT_MINCONN,
    'maxconn': DEFAULT_MAXCONN,
}

def set_global_db_config(**kwargs):
    """
    Set global database configuration parameters that will affect all subsequent get_db_config calls.
    
    Args:
        **kwargs: Database configuration parameters to set globally.
                 Valid keys: 'host', 'port', 'user', 'password', 'minconn', 'maxconn'
                 
    Examples:
        # Use localhost
        set_global_db_config(host='localhost')
        
        # Use custom container
        set_global_db_config(host='custom_container', port=5433)
        
        # Use different credentials
        set_global_db_config(user='custom_user', password='custom_pass')
    """
    global GLOBAL_CONFIG
    for key, value in kwargs.items():
        if key in GLOBAL_CONFIG:
            GLOBAL_CONFIG[key] = value
        else:
            raise ValueError(f"Invalid configuration key: {key}. Valid keys are: {list(GLOBAL_CONFIG.keys())}")

def reset_global_db_config():
    """
    Reset all global database configuration parameters to their default values.
    """
    global GLOBAL_CONFIG
    GLOBAL_CONFIG = {
        'host': DEFAULT_HOST,
        'port': DEFAULT_PORT,
        'user': DEFAULT_USER,
        'password': DEFAULT_PASSWORD,
        'minconn': DEFAULT_MINCONN,
        'maxconn': DEFAULT_MAXCONN,
    }

def get_db_config(host=None, port=None, user=None, password=None, minconn=None, maxconn=None):
    """
    Get database configuration.
    
    Args:
        host (str): If provided, overrides the global host setting
        port (int): If provided, overrides the global port setting
        user (str): If provided, overrides the global user setting
        password (str): If provided, overrides the global password setting
        minconn (int): If provided, overrides the global minconn setting
        maxconn (int): If provided, overrides the global maxconn setting
    
    Returns:
        dict: Database configuration dictionary with all settings
    
    Examples:
        # Get configuration with global settings
        config = get_db_config()
        
        # Override specific settings
        config = get_db_config(host='localhost', port=5433)
    """
    # Start with a copy of the global configuration
    config = GLOBAL_CONFIG.copy()
    
    # Override with any explicitly provided parameters
    if host is not None:
        config['host'] = host
    if port is not None:
        config['port'] = port
    if user is not None:
        config['user'] = user
    if password is not None:
        config['password'] = password
    if minconn is not None:
        config['minconn'] = minconn
    if maxconn is not None:
        config['maxconn'] = maxconn
        
    return config 
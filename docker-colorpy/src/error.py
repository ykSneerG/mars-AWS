def status_error(status_code: int, message: str) -> dict:
    
    if status_code == 400:
        message = f"Bad Request: {message}"
    elif status_code == 404:
        message = f"Not Found: {message}"
    elif status_code == 500:
        message = f"Internal Server Error: {message}"
    else:
        message = f"Unknown Error: {message}"
    
    return {
        "statusCode": status_code, 
        "body": {
            "error": message
            }
        }

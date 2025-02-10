from typing import Any, Dict


def parse_http_request(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r') as f:
        lines = f.readlines()
  
    method, path, _ = lines[0].split()
    headers = {}
    for idx, line in enumerate(lines[1:]):
        if line == '\n':
            break
        key, value = line.strip().split(': ')
        headers[key] = value
    
    body = "".join([x.strip() for x in lines[idx+2:]])
    
    return {
        "method": method,
        "path": path,
        "headers": headers,
        "body": body
    }

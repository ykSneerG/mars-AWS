import os

def save_bytes_to_file(data, filename, output_dir="./docker-profiler/src/_run/output"):
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        if data:
            with open(os.path.join(output_dir, filename), "wb") as f:
                f.write(data)
        else:
            print("Key 'bytes' not found in response['body']")
            
        # check if file is succesfully written
        if os.path.exists(os.path.join(output_dir, filename)):
            print(f"File {filename} successfully written to {output_dir}")

    except Exception as e:
        print(f"Error occurred: {e}")
import hashlib
from src.code.marsHelper import RandomId
from src.code.files.botox import Botox
from src.code.files.cgatsToJson import CgatsToJson
from src.handlers import BaseLambdaHandler

import json

class Files_CgatsToJson_Handler(BaseLambdaHandler):
    
    def handle(self):
        
        try:        
            ctj = CgatsToJson({ **self.event })
            
            jd = {
                "elapsed": self.get_elapsed_time(),
                "result": ctj.get_result,
            }
            
            return self.get_common_response(jd)
        
        except Exception as e:
            return self.get_error_response(str(e)) 
        

class File_UploadCgats_Handler(BaseLambdaHandler):
    
    def handle(self):
        
        filename = self.event.get("fileName", None)
        filecontent = self.event.get("fileContent", None)

        if filename is None or filename == "":
            return self.get_error_response("No filename provided")
        
        if filecontent is None or filecontent == "":
            return self.get_error_response("No file content provided")


        filecontent_hash = hashlib.sha256(filecontent.encode()).hexdigest()
        
        
        # ----->>> NEED TO BE A DYNAMO DB CHECK <<<------ !!!! ENHANCEMENT
       
        # # Check if file already exists, than return UPID
        # It's better to use None here since hash_exists() likely returns None when no match is found
        # This makes the code more explicit and follows Python conventions
        #fexists = Botox("mars-predicted-data").hash_exists(filecontent_hash)
        # if fexists is not None:          
        #     return self.get_common_response({
        #         "hash": filecontent_hash,
        #         "filename": filename,
        #         "message": "File already exists.",
        #         "UPID": fexists,
        #         "bytes": len(filecontent),
        #         "elapsed": self.get_elapsed_time()
        #     })
            
        # return {
        #     "fexists": fexists,
        # }
        
        # Object name
        object_name = RandomId.random_id() + "-UPLOAD"

        try:
            # Store in bucket -- CGATS        
            datastore_cgats = Botox("mars-predicted-data")
            datastore_cgats.update_metadata("FileName", filename)
            datastore_cgats.update_metadata("FileContentHash", filecontent_hash)
            datastore_cgats_result = datastore_cgats.store_S3(
                object_name, 
                filecontent,
                f"data/{object_name}.txt"
            )
        except Exception as e:
            return self.get_error_response(f"Error uploading to S3: {str(e)}")    
        
        jd = {
            "hash": filecontent_hash,
            # "fexists": fexists,
            "filename": filename,
            "message": "File uploaded successfully.",
            "UPID": datastore_cgats_result["UPID"],
            "bytes": datastore_cgats_result["bytes"],
            "elapsed": self.get_elapsed_time()
        }
                
        return self.get_common_response(jd)
    
    
    
class File_UploadHashExists_Handler(BaseLambdaHandler):
    pass



class File_UploadedToJson_Handler(BaseLambdaHandler):
    
    def handle(self):
        
        upload_id = self.event.get("uploadId", "")
        if upload_id == "":
            return self.get_error_response("No upload ID provided")
        
        try:
            datastore = Botox("mars-predicted-data")
        
            ctj = CgatsToJson({
                **self.event,
                "txt": datastore.load_S3(f"data/{upload_id}.txt")
            })
            
            jd = {
                "elapsed": self.get_elapsed_time(),
                "upload_id": upload_id,
                "result": ctj.get_result
            }
            
            return self.get_common_response(jd)
        
        except Exception as e:
                return self.get_error_response(str(e))    

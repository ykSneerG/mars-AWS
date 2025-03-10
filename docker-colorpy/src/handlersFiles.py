from src.code.marsHelper import RandomId
from src.code.files.botox import Botox
from src.code.files.cgatsToJson import CgatsToJson
from src.handlers import BaseLambdaHandler


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
        
        filename = self.event.get("fileName", "")
        filecontent = self.event.get("fileContent", "")

        if filename == "":
            return self.get_error_response("No filename provided")
        
        if filecontent == "":
            return self.get_error_response("No file content provided")

        jd = {}
        
        # Object name
        object_name = RandomId.random_id() + "-UPLOAD"

        try:
            # Store in bucket -- CGATS        
            datastore_cgats = Botox("mars-predicted-data")
            datastore_cgats_result = datastore_cgats.store_S3(
                object_name, 
                filecontent,
                f"data/{object_name}.txt"
            )
        except Exception as e:
            return self.get_error_response(str(e))    
        
        jd.update({
            "message": "File uploaded successfully.",
            "UPID": datastore_cgats_result["UPID"],
            "bytes": datastore_cgats_result["bytes"],
            "elapsed": self.get_elapsed_time()
        })
                
        return self.get_common_response(jd)
    

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
                "result": ctj.get_result,
            }
            
            return self.get_common_response(jd)
        
        except Exception as e:
                return self.get_error_response(str(e))    

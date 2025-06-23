from src.code.profile.profiler import Profiler


class Profile_Handler:
    def __init__(self, event, context):
        self.event = event
        self.context = context
        self.profiler = Profiler()
        
        """ 
        const body = {
            "src_dcs": dcs,
            "src_pcs": pcs,
            "dst_space": dstSpace
        }
        """
        self.dcs = event.get("src_dcs")
        self.pcs = event.get("src_pcs")
        #self.dst_space = event.get("dst_space")

    def handle(self):
        self.profiler.start()
        # Handle the event and context
        result = self.process_event()
        
        
        
        
        
        self.profiler.stop()
        elapsed_time = self.profiler.get_elapsed_time()
        return {
            "result": result,
            "elapsed_time": elapsed_time
        }

    def process_event(self):
        # Process the event and context
        return "Processed event"
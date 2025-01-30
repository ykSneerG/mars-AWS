from typing import Any
from src.handlers import BaseLambdaHandler
from src.code.predict.linearization.synlinV4a import SynLinSolidV4a


class Predict_SynlinV4_Handler(BaseLambdaHandler):
    
    def handle(self):
        
        jd = {}
    
        media = self.event['media']
        solid = self.event['solid']
        steps = int(self.event.get('steps', 5))
        iters = int(self.event.get('iterations', 1)) 
        toler = float(self.event.get('tolerance', 0.004))
        preci = int(self.event.get('precision', 3))
        """ cfact = float(self.event.get('correction', 4.5)) """

        sls = SynLinSolidV4a()
        """ sls.set_places = int(event.get('round', 3))  """
        sls.set_precision(preci)
        
        sls.set_media(media)
        sls.set_solid(solid)

        err = sls.set_gradient_by_steps(steps)
        if err: return self.get_error_response(err)

        #sls.set_gradient([0.0, 50.0, 100.0])
        sls.tolerance = toler
        sls.set_max_loops(iters)
        """ sls.setCorrection(cfact) """
        res: dict[str, Any] = sls.start()
        
        jd.update(res)   

        jd.update({
            'elapsed': self.get_elapsed_time()
        })
        return self.get_common_response(jd)

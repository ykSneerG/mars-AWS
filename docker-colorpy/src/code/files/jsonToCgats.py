from datetime import datetime

class JsonToCgats:
    
    def __init__(self, json_data):
        
        self.originator = "ColorSpace.cloud"
        
        self.json_data = json_data
    
    def convert(self):

        # --- DATA_FORMAT ---
        data_format_sid = ['SAMPLE_ID']
        data_format_pcs = [f'SPECTRAL_NM_{380 + i * 10}' for i in range(36)]
        dcs_len = len(self.json_data[0]['dcs'])
        data_format_dcs = [f'PC{dcs_len}_{i+1}' for i in range(dcs_len)]

        dataformat = data_format_sid + data_format_dcs + data_format_pcs

        # --- DATA ---
        data = []
        for i, item in enumerate(self.json_data):
            sid = i + 1
            dcs = [f"{x:.2f}" for x in item['dcs']]
            snm = [f"{x:.4f}" for x in item['snm']]
            data.append('\t'.join(map(str, [sid] + dcs + snm)))

        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z")

        header = []
        header.append('CGATS.17')
        header.append(f'ORIGINATOR\t"{self.originator}"')
        header.append(f'CREATED\t"{timestamp}"')
        header.append('FILE_DESCRIPTOR\t"Synthetic data using the MarsV4a model."')
        header.append('KEYWORD\t"ILLUMINATION_NAME"')
        header.append('ILLUMINATION_NAME\t"D50"')
        header.append('KEYWORD\t"OBSERVER_ANGLE"')
        header.append('OBSERVER_ANGLE\t"2"')
        header.append('KEYWORD\t"MEASURE_CONDITION"')
        header.append('MEASURE_CONDITION\t"M1"')
        header.append(f'NUMBER_OF_FIELDS\t{len(dataformat)}')
        header.append('BEGIN_DATA_FORMAT')
        header.append('\t'.join(dataformat))
        header.append('END_DATA_FORMAT')
        header.append(f'NUMBER_OF_SETS\t{len(data)}')
        header.append('BEGIN_DATA')

        footer = ['END_DATA']

        result = '\n'.join(header + data + footer)

        return result.encode('utf-8')


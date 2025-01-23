import datetime
from src.code.colorMath import CmMath
from src.code.files.cgatsConstants import CgatsConstants


class CgatsCreator:
    def __init__(self) -> None:
        self.result = CgatsConstants.UNSET

    def set_json(self, json_data: dict):
        self.result = json_data

    def get_txt(self):
        tmp = ["CGATS.17"]

        # ----------------------------------------------------

        keys = [
            ("originator", "ORIGINATOR"),
            ("descriptor", "FILE_DESCRIPTOR"),
            ("instrument", "INSTRUMENT")
        ]

        for k in keys:
            self.__check_key_head(tmp, k[0], k[1])

        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%dT%H:%M:%S")
        tmp.append("CREATED" + "\t\"" + formatted_datetime + "\"")

        if "lgorowlength" in self.result["head"]:
            tmp.append("KEYWORD \"LGOROWLENGTH\"")
            tmp.append("LGOROWLENGTH\t" + str(self.result["head"]["lgorowlength"]))

        '''
        --ORIGINATOR "ColorGATE"
        --FILE_DESCRIPTOR "Unknown"
        --CREATED "2023-04-27T10:15:40"
        CMYK-Advanced	"Target"

        KEYWORD "LGOROWLENGTH"
        LGOROWLENGTH	21

        --DESCRIPTOR	"CMYK-Advanced Target Solvent"

        KEYWORD	"INFO"
        INFO	"Linearization target for solvent printers with an accurate characteristic in the ranges 0..30% and 80..100%."

        --INSTRUMENTATION	"Barbieri Spectro LFP"

        KEYWORD	"ILLUMINATION_NAME"
        ILLUMINATION_NAME	"D50"

        KEYWORD	"OBSERVER_ANGLE"
        OBSERVER_ANGLE	"2"

        --NUMBER_OF_FIELDS 9
        '''
        tmp.append("NUMBER_OF_FIELDS" + "\t" + str(self.__count_num_fields(self.result["data"])))

        # ----------------------------------------------------

        tmp.append("BEGIN_DATA_FORMAT")
        '''
        SAMPLE_ID	SAMPLE_NAME	CMYK_Y	CMYK_M	CMYK_C	CMYK_K	LAB_L	LAB_A	LAB_B
        '''

        keys: list[str] = list(self.result["data"][0].keys())

        if "id" in keys:
            keys = CgatsCreator.replace_list_element_with_list(keys, "id", ["SAMPLE_ID"])

        if "name" in keys:
            keys = CgatsCreator.replace_list_element_with_list(keys, "name", ["SAMPLE_NAME"])

        if "qty" in keys:
            keys = CgatsCreator.replace_list_element_with_list(keys, "qty", ["SAMPLE_QTY"])

        # - : - : - : - : - : - : - : - : - : - : - : - : - : - : - : - : - : - : - DCS

        tpl_dcs = CgatsConstants.find_tuple_by_string(
            CgatsConstants.get_list_dcs(),
            self.result["typeDCS"]
            )

        if tpl_dcs is not None:
            keys = CgatsCreator.replace_list_element_with_list(keys, "dcs", tpl_dcs[1])

        # - : - : - : - : - : - : - : - : - : - : - : - : - : - : - : - : - : - : - PCS

        tpl_pcs = CgatsConstants.find_tuple_by_string(
            CgatsConstants.get_list_pcs(),
            self.result["typePCS"]
            )

        if tpl_pcs is not None:
            keys = CgatsCreator.replace_list_element_with_list(keys, "pcs", tpl_pcs[1])

        # - - - - - - - - - - - - - - - - - - - - - - - - - -

        keys = "\t".join(list(keys))

        tmp.append(keys)

        tmp.append("END_DATA_FORMAT")

        # ----------------------------------------------------

        count = len(self.result["data"])
        tmp.append("NUMBER_OF_SETS" + "\t" + str(count))

        # ----------------------------------------------------

        tmp.append("BEGIN_DATA")

        for item in self.result["data"]:
            str_line = self.__item_to_row(item)
            tmp.append(str_line)

        tmp.append("END_DATA")

        # ----------------------------------------------------

        result = "\n".join(tmp)
        return result

    def __check_key_head(self, tmp: list, strCheck, strKey):
        if strCheck in self.result["head"]:
            # tmp_a = strKey + "\t\"" + self.result["head"][strCheck] + "\""
            tmp_a = f"{strKey}\t\"{self.result['head'][strCheck]}\""
            tmp.append(tmp_a)

    def __item_to_row(self, item: dict) -> str:
        values = []
        for value in item.values():
            if isinstance(value, list):
                values.append("\t".join(map(str, value)))
            else:
                values.append(str(value))

        str_line = "\t".join(values)
        return str_line

    def __count_num_fields(self, item: list[dict]):
        values = []
        for element in item:
            total = 0
            for value in element.values():
                if isinstance(value, list):
                    total += len(value)
                else:
                    total += 1

            values.append(total)

        if CmMath.check_same_value(values):
            return values[0]
        else:
            return False

    @staticmethod
    def replace_list_element_with_list(lst: list[str], search_string: str, replacement_list: list[str]):
        for i in range(len(lst)):
            if lst[i] == search_string:
                return lst[:i] + replacement_list + lst[i + 1:]

        return lst

    '''
    def get_csv(self):
        return NotImplementedError()
    '''

import re
import random
from src.code.files.cgatsConstants import CgatsConstants
from src.code.colorMath import CmMath


class CgatsHelper:
    @staticmethod
    def get_index(content, pattern):
        return [i for i, s in enumerate(content) if pattern.strip() == s.strip()][0]

    @staticmethod
    def responseEntry(tmp, value, name):
        if value != CgatsConstants.UNSET:
            tmp[name] = value

    @staticmethod
    def responseEntryCheck(tmp, check, value, name):
        if check != CgatsConstants.UNSET:
            tmp[name] = value


class CgatsDataObject:
    def __init__(self) -> None:
        self.type = CgatsConstants.UNSET
        self.keys = []

    def is_unset(self) -> bool:
        return self.type == CgatsConstants.UNSET

    def update(self, type, keys):
        self.type = type
        self.keys = keys

    def to_json(self):
        return {"type": self.type, "keys": self.keys}


class CgatsDataFormat:
    def __init__(self, line: str):
        self.lines = line.split()

        self.sDcs = CgatsDataObject()
        self.sPcs = CgatsDataObject()
        self.sId = CgatsDataObject()
        self.sName = CgatsDataObject()
        self.sQty = CgatsDataObject()

        self.__find_id()
        self.__find_name()
        self.__find_pcs()
        self.__find_dcs()
        self.__find_qty()

    def __find_id(self) -> None:
        self.__find_tpls(CgatsConstants.ID_TPLS, self.sId)

    def __find_name(self) -> None:
        self.__find_tpls(CgatsConstants.NAME_TPLS, self.sName)

    def __find_qty(self) -> None:
        self.__find_tpls(CgatsConstants.QTY_TPLS, self.sQty)

    def __find_pcs(self) -> None:
        for key in CgatsConstants.get_list_pcs():
            self.__find_tpls(key, self.sPcs)

    def __find_dcs(self) -> None:
        for key in CgatsConstants.get_list_dcs():
            self.__find_tpls(key, self.sDcs)

    def __find_keys(self, searchKeys, info, sObj: CgatsDataObject) -> None:
        if not sObj.is_unset():
            return

        found = False
        ident = []

        for skey in searchKeys:
            found = skey in self.lines

            if found:
                ident.append(self.lines.index(skey))
            else:
                ident.clear()
                break

        if found:
            sObj.update(info, ident)

    def __find_tpls(self, ski: tuple, sObj: CgatsDataObject) -> None:
        self.__find_keys(ski[1], ski[0], sObj)

    def get_type_dcs(self):
        return self.sDcs.type

    def get_type_pcs(self):
        return self.sPcs.type

    def to_json(self) -> dict:
        tmp = {}

        CgatsHelper.responseEntryCheck(tmp, self.sId.type, self.sId.to_json(), "id")
        CgatsHelper.responseEntryCheck(tmp, self.sName.type, self.sName.to_json(), "name")
        CgatsHelper.responseEntryCheck(tmp, self.sDcs.type, self.sDcs.to_json(), "dcs")
        CgatsHelper.responseEntryCheck(tmp, self.sPcs.type, self.sPcs.to_json(), "pcs")
        CgatsHelper.responseEntryCheck(tmp, self.sQty.type, self.sQty.to_json(), "qty")

        return tmp


class CgatsDataTable:
    def __init__(self, lines: list[str], df: dict) -> None:
        self.lines = lines
        self.df = df
        self.result = [self.__line_to_object(line.split("\t")) for line in self.lines]

    def __line_to_object(self, splitted: list[str]) -> dict:
        row_data = {}
        self.__get_data(splitted, row_data, "id")
        self.__get_data(splitted, row_data, "name")
        self.__get_data(splitted, row_data, "dcs", True)
        self.__get_data(splitted, row_data, "pcs", True)
        self.__get_data(splitted, row_data, "qty", True)
        self.__get_data(splitted, row_data, "cat")
        return row_data

    def __get_data(self, spl: str, row_data: dict, x: str, convert: bool = False) -> None:
        if x in self.df:
            keys = self.df[x]["keys"]
            if len(keys) == 1:
                key = keys[0]
                row_data[x] = CmMath.toNum(spl[key]) if convert else spl[key]
            else:
                row_data[x] = [CmMath.toNum(spl[elem]) if convert else spl[elem] for elem in keys]

    def to_json(self) -> list[dict]:
        return self.result


class CgatsHead:
    def __init__(self, head_lines: list[str]):
        self.head_lines = head_lines

        self.default_words = {
            "Lgorowlength": self.__find_entry(CgatsConstants.LGOROWLENGTH),
            "Originator": self.__find_entry(CgatsConstants.ORIGINATOR),
            "Descriptor": self.__find_entry(CgatsConstants.FILE_DESCRIPTOR),
            "Creation": self.__find_entry(CgatsConstants.CREATED),
            "Instrument": self.__find_entry(CgatsConstants.INSTRUMENTATION),
            "measurementGeometry": self.__find_entry(CgatsConstants.MEASUREMENT_GEOMETRY),
            "measurementSource": self.__find_entry(CgatsConstants.MEASUREMENT_SOURCE),
            "filter": self.__find_entry(CgatsConstants.FILTER),
            "polarization": self.__find_entry(CgatsConstants.POLARIZATION),
            "weightningFunction": self.__find_entry(CgatsConstants.WEIGHTING_FUNCTION),
            "computationalParameter": self.__find_entry(CgatsConstants.COMPUTATIONAL_PARAMETER),
            "sampleBacking": self.__find_entry(CgatsConstants.SAMPLE_BACKING),
            "manufactor": self.__find_entry(CgatsConstants.MANUFACTURER),
            "material": self.__find_entry(CgatsConstants.MATERIAL),
            "targetType": self.__find_entry(CgatsConstants.TARGET_TYPE),
            "colorant": self.__find_entry(CgatsConstants.COLORANT),
            "prodDate": self.__find_entry(CgatsConstants.PROD_DATE),
            "printConditions": self.__find_entry(CgatsConstants.PRINT_CONDITIONS),
            "serial": self.__find_entry(CgatsConstants.SERIAL),
            "processcolorId": self.__find_entry(CgatsConstants.PROCESSCOLOR_ID),
            "spotId": self.__find_entry(CgatsConstants.SPOT_ID)
        }
        self.custom_words = self.__find_keyword()

    def __find_keyword(self) -> dict:

        keyword_values = {}
        current_keyword = None

        for line in self.head_lines:
            if line.strip().startswith('KEYWORD'):
                current_keyword = line.split('"')[1].strip()
                keyword_values[current_keyword] = None

        for key in keyword_values:
            for line in self.head_lines:
                if line.startswith(key):
                    value = line.split('\t')[1].strip("\"")
                    keyword_values[key] = CmMath.toNum(value)

        return keyword_values

    def __find_entry(self, keywords):
        for entry in self.head_lines:
            for keyword in keywords:
                if keyword in entry:
                    opt_line = entry.strip()
                    # Check if the line has a value in quotes
                    m = re.search('"([^"]+)"', opt_line)
                    if m:
                        result = m.group(1)
                        return CmMath.toNum(result)
                    # If no value in quotes, get the last word in the line
                    else:
                        result = opt_line.split()[-1]
                        return CmMath.toNum(result)
        return CgatsConstants.UNSET

    def to_json(self) -> dict:
        tmp = {**self.default_words, **self.custom_words}
        tmp = {k.lower(): v for k, v in tmp.items() if v is not None and v != CgatsConstants.UNSET and v != "Unknown"}
        return tmp


class Cgats:
    def __init__(self, fileContent: list[str]) -> None:
        idxBeginData = CgatsHelper.get_index(fileContent, CgatsConstants.BEGIN_DATA)
        idxEndData = CgatsHelper.get_index(fileContent, CgatsConstants.END_DATA)

        idxBeginDataFormat = CgatsHelper.get_index(
            fileContent, CgatsConstants.BEGIN_DATA_FORMAT
        )
        # idxEndDataFormat   = Cgats.get_index(fileContent, CgatsConstants.END_DATA_FORMAT)

        # -----
        df = fileContent[idxBeginDataFormat + 1]
        self.dataFormat = CgatsDataFormat(df)
        # -----
        table = fileContent[idxBeginData+1:idxEndData]
        self.dataTable = CgatsDataTable(table, self.dataFormat.to_json())
        # -----
        headl = fileContent[0:idxBeginData]
        self.head = CgatsHead(headl)

    def get_header(self) -> dict:
        return self.head.to_json()

    def set_data(self, data) -> None:
        self.dataTable.result = data

    def get_type_dcs(self):
        return self.dataFormat.get_type_dcs()

    def get_type_pcs(self):
        return self.dataFormat.get_type_pcs()

    def row_random(self):
        return random.choice(self.dataTable.result)

    def row_id(self, id: int):
        return self.dataTable.result[id] if 0 <= id < len(self.dataTable.result) else None

    def row_range(self, rangeMin: list, rangeMax: list):
        return [item for item in self.dataTable.result if CmMath.inRange(item["dcs"], rangeMin, rangeMax)]

    def row_all(self):
        return self.dataTable.result

    def dcs_order(self, order: list[int]):

        dcs_length = len(self.dataTable.result[0]["dcs"])

        if len(order) == dcs_length:
            self.dataTable.result = self.__swap_dcs(order)

        if len(order) < dcs_length:
            self.dataTable.result = self.__remove_dcs(order)
            self.dataTable.result = self.__swap_dcs(order)

        return self.dataTable.result

    def __swap_dcs(self, order: list[int]):
        dcs_length = len(self.dataTable.result[0]["dcs"])

        if any(elem not in range(dcs_length) for elem in order):
            return self.dataTable.result

        return [
            {**item, "dcs": [item["dcs"][i] for i in order]}
            for item in self.dataTable.result
        ]

    def __remove_dcs(self, order: list[int]):
        indices_actual: set[int] = set(range(len(self.dataTable.result[0]["dcs"])))
        indices_tokeep: set[int] = set(i for i in range(len(self.dataTable.result[0]["dcs"])) if i in order)
        indices_delete: set[int] = indices_actual - indices_tokeep

        lst_removed = []
        for item in self.dataTable.result:
            append = False
            for i in indices_delete:
                append = item["dcs"][i] != 0.0
                if append:
                    break
            if not append:
                lst_removed.append(item)

        return lst_removed

    def to_json(self):
        res = {}
        res['head'] = self.get_header()
        res['typeDCS'] = self.get_type_dcs()
        res['typePCS'] = self.get_type_pcs()
        res['data'] = self.row_all()
        return res

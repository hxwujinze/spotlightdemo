import os,base64
from PIL import Image
import io
def decode_base64(data):
            """Decode base64, padding being optional.
            :param data: Base64 data as an ASCII byte string
            :returns: The decoded byte string.
            """
            missing_padding = 4 - len(data) % 4
            if missing_padding:
                data += b'='* missing_padding
            return base64.decodestring(data)
leniystr =b'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'
imgdata=decode_base64(leniystr)
image = io.BytesIO(imgdata)
img = Image.open(image)
img.show()

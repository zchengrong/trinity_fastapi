import os

DESIGN_TRITON_IP = os.getenv('DESIGN_TRITON_IP', '10.1.1.240')
DESIGN_TRITON_PORT = os.getenv('DESIGN_TRITON_PORT', '8000')
SEGMENTATION = os.getenv("SEGMENTATION", {
    "name": "seg_ocrnet_hr18",
    "input": "seg_input__0",
    "output": "seg_output__0",
})

GENERATE_TRITON_IP = os.getenv('TRITON_IP_DEFAULT', '10.1.1.240')
GENERATE_TRITON_PORT_HTTP = os.getenv('GENERATE_TRITON_PORT_HTTP', '7000')
GENERATE_TRITON_PORT_gPRC = os.getenv('GENERATE_TRITON_PORT_gPRC', '7001')

from __future__ import annotations

import ctypes
import time
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from roc.mvs.sdk_loader import ensure_mvs_python_path

ensure_mvs_python_path()

from CameraParams_const import MV_ACCESS_Exclusive  # type: ignore  # noqa: E402
from CameraParams_const import MV_GENTL_CAMERALINK_DEVICE  # type: ignore  # noqa: E402
from CameraParams_const import MV_GENTL_CXP_DEVICE  # type: ignore  # noqa: E402
from CameraParams_const import MV_GENTL_GIGE_DEVICE  # type: ignore  # noqa: E402
from CameraParams_const import MV_GENTL_XOF_DEVICE  # type: ignore  # noqa: E402
from CameraParams_const import MV_GIGE_DEVICE  # type: ignore  # noqa: E402
from CameraParams_const import MV_USB_DEVICE  # type: ignore  # noqa: E402
from CameraParams_header import MV_CC_DEVICE_INFO  # type: ignore  # noqa: E402
from CameraParams_header import MV_CC_DEVICE_INFO_LIST  # type: ignore  # noqa: E402
from CameraParams_header import MV_CC_PIXEL_CONVERT_PARAM_EX  # type: ignore  # noqa: E402
from CameraParams_header import MV_FRAME_OUT  # type: ignore  # noqa: E402
from CameraParams_header import MV_TRIGGER_MODE_ON  # type: ignore  # noqa: E402
from MvCameraControl_class import MvCamera  # type: ignore  # noqa: E402
from MvErrorDefine_const import MV_E_NODATA  # type: ignore  # noqa: E402
from MvErrorDefine_const import MV_OK  # type: ignore  # noqa: E402
from PixelType_header import PixelType_Gvsp_BGR8_Packed  # type: ignore  # noqa: E402


DEVICE_MASK = (
    MV_GIGE_DEVICE
    | MV_USB_DEVICE
    | MV_GENTL_GIGE_DEVICE
    | MV_GENTL_CAMERALINK_DEVICE
    | MV_GENTL_CXP_DEVICE
    | MV_GENTL_XOF_DEVICE
)


def _decode_char_array(values: Iterable[int]) -> str:
    chars = []
    for value in values:
        if value == 0:
            break
        chars.append(chr(value))
    return "".join(chars)


def _transport_name(transport_type: int) -> str:
    if transport_type in (MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE):
        return "GigE"
    if transport_type == MV_USB_DEVICE:
        return "USB3"
    if transport_type == MV_GENTL_CAMERALINK_DEVICE:
        return "CameraLink"
    if transport_type == MV_GENTL_CXP_DEVICE:
        return "CXP"
    if transport_type == MV_GENTL_XOF_DEVICE:
        return "XoF"
    return "Unknown"


@dataclass(slots=True)
class CameraInfo:
    index: int
    serial: str
    model_name: str
    transport_type: int
    transport_name: str
    width: int = 0
    height: int = 0


class MvsError(RuntimeError):
    pass


class MvsSystem:
    def __init__(self) -> None:
        ret = MvCamera.MV_CC_Initialize()
        if ret != MV_OK:
            raise MvsError(f"MV_CC_Initialize failed: 0x{ret:x}")
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            MvCamera.MV_CC_Finalize()
            self._closed = True

    def __enter__(self) -> "MvsSystem":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def enumerate_devices(self) -> list[CameraInfo]:
        device_list = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(DEVICE_MASK, device_list)
        if ret != MV_OK:
            raise MvsError(f"MV_CC_EnumDevices failed: 0x{ret:x}")

        devices: list[CameraInfo] = []
        for index in range(device_list.nDeviceNum):
            dev_info = ctypes.cast(
                device_list.pDeviceInfo[index],
                ctypes.POINTER(MV_CC_DEVICE_INFO),
            ).contents

            if dev_info.nTLayerType in (MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE):
                special = dev_info.SpecialInfo.stGigEInfo
                serial = _decode_char_array(special.chSerialNumber)
                model_name = _decode_char_array(special.chModelName)
            elif dev_info.nTLayerType == MV_USB_DEVICE:
                special = dev_info.SpecialInfo.stUsb3VInfo
                serial = _decode_char_array(special.chSerialNumber)
                model_name = _decode_char_array(special.chModelName)
            elif dev_info.nTLayerType == MV_GENTL_CAMERALINK_DEVICE:
                special = dev_info.SpecialInfo.stCMLInfo
                serial = _decode_char_array(special.chSerialNumber)
                model_name = _decode_char_array(special.chModelName)
            elif dev_info.nTLayerType == MV_GENTL_CXP_DEVICE:
                special = dev_info.SpecialInfo.stCXPInfo
                serial = _decode_char_array(special.chSerialNumber)
                model_name = _decode_char_array(special.chModelName)
            else:
                special = dev_info.SpecialInfo.stXoFInfo
                serial = _decode_char_array(special.chSerialNumber)
                model_name = _decode_char_array(special.chModelName)

            devices.append(
                CameraInfo(
                    index=index,
                    serial=serial,
                    model_name=model_name,
                    transport_type=dev_info.nTLayerType,
                    transport_name=_transport_name(dev_info.nTLayerType),
                )
            )
        return devices

    def open_camera(self, camera_index: int) -> "MvsCamera":
        device_list = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(DEVICE_MASK, device_list)
        if ret != MV_OK:
            raise MvsError(f"MV_CC_EnumDevices failed: 0x{ret:x}")
        if camera_index >= device_list.nDeviceNum:
            raise MvsError(f"Camera index out of range: {camera_index}")

        dev_info = ctypes.cast(
            device_list.pDeviceInfo[camera_index],
            ctypes.POINTER(MV_CC_DEVICE_INFO),
        ).contents
        camera = MvsCamera(dev_info, camera_index)
        camera.open()
        return camera


class MvsCamera:
    def __init__(self, dev_info: MV_CC_DEVICE_INFO, camera_index: int) -> None:
        self._dev_info = dev_info
        self._camera_index = camera_index
        self._camera = MvCamera()
        self._opened = False
        self._grabbing = False

        if dev_info.nTLayerType in (MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE):
            special = dev_info.SpecialInfo.stGigEInfo
            self.serial = _decode_char_array(special.chSerialNumber)
            self.model_name = _decode_char_array(special.chModelName)
        elif dev_info.nTLayerType == MV_USB_DEVICE:
            special = dev_info.SpecialInfo.stUsb3VInfo
            self.serial = _decode_char_array(special.chSerialNumber)
            self.model_name = _decode_char_array(special.chModelName)
        elif dev_info.nTLayerType == MV_GENTL_CAMERALINK_DEVICE:
            special = dev_info.SpecialInfo.stCMLInfo
            self.serial = _decode_char_array(special.chSerialNumber)
            self.model_name = _decode_char_array(special.chModelName)
        elif dev_info.nTLayerType == MV_GENTL_CXP_DEVICE:
            special = dev_info.SpecialInfo.stCXPInfo
            self.serial = _decode_char_array(special.chSerialNumber)
            self.model_name = _decode_char_array(special.chModelName)
        else:
            special = dev_info.SpecialInfo.stXoFInfo
            self.serial = _decode_char_array(special.chSerialNumber)
            self.model_name = _decode_char_array(special.chModelName)
        self.transport_type = dev_info.nTLayerType
        self.transport_name = _transport_name(dev_info.nTLayerType)

    def open(self) -> None:
        ret = self._camera.MV_CC_CreateHandle(self._dev_info)
        if ret != MV_OK:
            raise MvsError(f"[{self.serial}] create handle failed: 0x{ret:x}")

        ret = self._camera.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != MV_OK:
            self._camera.MV_CC_DestroyHandle()
            raise MvsError(f"[{self.serial}] open device failed: 0x{ret:x}")

        self._opened = True
        if self.transport_type in (MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE):
            packet_size = self._camera.MV_CC_GetOptimalPacketSize()
            if int(packet_size) > 0:
                self._camera.MV_CC_SetIntValue("GevSCPSPacketSize", packet_size)

    def close(self) -> None:
        try:
            if self._grabbing:
                self.stop_grabbing()
        finally:
            if self._opened:
                self._camera.MV_CC_CloseDevice()
                self._opened = False
            self._camera.MV_CC_DestroyHandle()

    def set_enum_by_string(self, key: str, value: str) -> None:
        ret = self._camera.MV_CC_SetEnumValueByString(key, value)
        if ret != MV_OK:
            raise MvsError(f"[{self.serial}] set {key}={value} failed: 0x{ret:x}")

    def set_enum(self, key: str, value: int) -> None:
        ret = self._camera.MV_CC_SetEnumValue(key, value)
        if ret != MV_OK:
            raise MvsError(f"[{self.serial}] set {key}={value} failed: 0x{ret:x}")

    def set_float(self, key: str, value: float) -> None:
        ret = self._camera.MV_CC_SetFloatValue(key, value)
        if ret != MV_OK:
            raise MvsError(f"[{self.serial}] set {key}={value} failed: 0x{ret:x}")

    def get_float(self, key: str) -> float:
        value = MVCC_FLOATVALUE()
        ret = self._camera.MV_CC_GetFloatValue(key, value)
        if ret != MV_OK:
            raise MvsError(f"[{self.serial}] get {key} failed: 0x{ret:x}")
        return float(value.fCurValue)

    def set_bool(self, key: str, value: bool) -> None:
        ret = self._camera.MV_CC_SetBoolValue(key, value)
        if ret != MV_OK:
            raise MvsError(f"[{self.serial}] set {key}={value} failed: 0x{ret:x}")

    def set_command(self, key: str) -> None:
        ret = self._camera.MV_CC_SetCommandValue(key)
        if ret != MV_OK:
            raise MvsError(f"[{self.serial}] command {key} failed: 0x{ret:x}")

    def set_pixel_format(self, pixel_format: str) -> None:
        self.set_enum_by_string("PixelFormat", pixel_format)

    def set_trigger_software_mode(self) -> None:
        self.set_enum("TriggerMode", MV_TRIGGER_MODE_ON)
        try:
            self.set_enum_by_string("TriggerSource", "Software")
            return
        except MvsError:
            pass

        # Fallback for cameras/SDK variants that do not accept symbolic strings here.
        self.set_enum("TriggerSource", 7)

    def disable_auto_exposure(self) -> None:
        self.set_enum_by_string("ExposureAuto", "Off")

    def disable_auto_gain(self) -> None:
        self.set_enum_by_string("GainAuto", "Off")

    def apply_manual_capture(self, exposure_us: float, gain_db: float, pixel_format: str) -> None:
        try:
            self.set_pixel_format(pixel_format)
        except MvsError:
            pass
        self.disable_auto_exposure()
        self.disable_auto_gain()
        self.set_float("ExposureTime", exposure_us)
        self.set_float("Gain", gain_db)
        self.set_trigger_software_mode()

    def start_grabbing(self) -> None:
        ret = self._camera.MV_CC_StartGrabbing()
        if ret != MV_OK:
            raise MvsError(f"[{self.serial}] start grabbing failed: 0x{ret:x}")
        self._grabbing = True

    def stop_grabbing(self) -> None:
        ret = self._camera.MV_CC_StopGrabbing()
        if ret != MV_OK:
            raise MvsError(f"[{self.serial}] stop grabbing failed: 0x{ret:x}")
        self._grabbing = False

    def trigger_software(self) -> None:
        self.set_command("TriggerSoftware")

    def grab_frame(self, timeout_ms: int = 1000) -> np.ndarray | None:
        frame = MV_FRAME_OUT()
        ctypes.memset(ctypes.byref(frame), 0, ctypes.sizeof(frame))
        ret = self._camera.MV_CC_GetImageBuffer(frame, timeout_ms)
        if ret == MV_E_NODATA:
            return None
        if ret != MV_OK:
            raise MvsError(f"[{self.serial}] get image buffer failed: 0x{ret:x}")

        try:
            width = int(frame.stFrameInfo.nWidth)
            height = int(frame.stFrameInfo.nHeight)
            dst_size = width * height * 3
            dst_buffer = (ctypes.c_ubyte * dst_size)()

            convert_param = MV_CC_PIXEL_CONVERT_PARAM_EX()
            ctypes.memset(ctypes.byref(convert_param), 0, ctypes.sizeof(convert_param))
            convert_param.nWidth = width
            convert_param.nHeight = height
            convert_param.pSrcData = frame.pBufAddr
            convert_param.nSrcDataLen = frame.stFrameInfo.nFrameLen
            convert_param.enSrcPixelType = frame.stFrameInfo.enPixelType
            convert_param.enDstPixelType = PixelType_Gvsp_BGR8_Packed
            convert_param.pDstBuffer = ctypes.cast(dst_buffer, ctypes.POINTER(ctypes.c_ubyte))
            convert_param.nDstBufferSize = dst_size

            ret = self._camera.MV_CC_ConvertPixelTypeEx(convert_param)
            if ret != MV_OK:
                raise MvsError(f"[{self.serial}] convert pixel failed: 0x{ret:x}")

            image = np.ctypeslib.as_array(dst_buffer).reshape(height, width, 3).copy()
            return image
        finally:
            self._camera.MV_CC_FreeImageBuffer(frame)

    def snapshot(self, fps_sleep: float = 0.0) -> np.ndarray | None:
        self.trigger_software()
        if fps_sleep > 0:
            time.sleep(fps_sleep)
        return self.grab_frame()

    def __enter__(self) -> "MvsCamera":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


from CameraParams_header import MVCC_FLOATVALUE  # type: ignore  # noqa: E402

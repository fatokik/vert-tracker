"""Application configuration via Pydantic Settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DroneSettings(BaseSettings):
    """Tello drone connection settings."""

    model_config = SettingsConfigDict(env_prefix="TELLO_")

    ip: str = "192.168.10.1"
    command_port: int = 8889
    state_port: int = 8890
    video_port: int = 11111
    connect_timeout: float = 10.0
    hover_height_cm: int = 100


class PoseSettings(BaseSettings):
    """MediaPipe pose estimation settings."""

    model_config = SettingsConfigDict(env_prefix="POSE_")

    model_complexity: Literal[0, 1, 2] = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    enable_segmentation: bool = False


class JumpDetectionSettings(BaseSettings):
    """Jump detection algorithm parameters."""

    model_config = SettingsConfigDict(env_prefix="JUMP_")

    takeoff_velocity_threshold: float = -8.0
    landing_velocity_threshold: float = 8.0
    min_airborne_frames: int = 5
    max_airborne_frames: int = 60
    landing_stability_frames: int = 3


class FilterSettings(BaseSettings):
    """Kalman filter and smoothing parameters."""

    model_config = SettingsConfigDict(env_prefix="")

    kalman_process_noise: float = Field(default=0.01, alias="KALMAN_PROCESS_NOISE")
    kalman_measurement_noise: float = Field(default=0.1, alias="KALMAN_MEASUREMENT_NOISE")
    smoothing_window_size: int = Field(default=5, alias="SMOOTHING_WINDOW_SIZE")


class CalibrationSettings(BaseSettings):
    """Calibration system settings."""

    model_config = SettingsConfigDict(env_prefix="")

    aruco_dict: str = Field(default="DICT_4X4_50", alias="ARUCO_DICT")
    aruco_marker_size_cm: float = Field(default=15.0, alias="ARUCO_MARKER_SIZE_CM")
    default_px_per_cm: float = Field(default=5.0, alias="DEFAULT_PX_PER_CM")
    calibration_distance_cm: float = Field(default=250.0, alias="CALIBRATION_DISTANCE_CM")


class UISettings(BaseSettings):
    """Display and overlay settings."""

    model_config = SettingsConfigDict(env_prefix="")

    display_width: int = Field(default=1280, alias="DISPLAY_WIDTH")
    display_height: int = Field(default=720, alias="DISPLAY_HEIGHT")
    show_skeleton: bool = Field(default=True, alias="SHOW_SKELETON")
    show_trajectory: bool = Field(default=True, alias="SHOW_TRAJECTORY")
    show_metrics: bool = Field(default=True, alias="SHOW_METRICS")
    show_debug_info: bool = Field(default=False, alias="SHOW_DEBUG_INFO")


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    model_config = SettingsConfigDict(env_prefix="LOG_")

    level: str = "INFO"
    file: str | None = None


class Settings(BaseSettings):
    """Root settings aggregating all configuration sections."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    drone: DroneSettings = Field(default_factory=DroneSettings)
    pose: PoseSettings = Field(default_factory=PoseSettings)
    jump: JumpDetectionSettings = Field(default_factory=JumpDetectionSettings)
    filter: FilterSettings = Field(default_factory=FilterSettings)
    calibration: CalibrationSettings = Field(default_factory=CalibrationSettings)
    ui: UISettings = Field(default_factory=UISettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings instance."""
    return Settings()

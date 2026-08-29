"""Tests for FlightSession RC control and safe shutdown."""

from __future__ import annotations

from vert_tracker.drone.flight_session import FlightSession
from vert_tracker.ui.display import KeyAction


class FakeController:
    """Minimal drone controller fake for FlightSession tests."""

    def __init__(self) -> None:
        self.rc_calls: list[tuple[int, int, int, int]] = []
        self.takeoff_calls = 0
        self.land_calls = 0
        self.battery_calls = 0
        self.stop_stream_calls = 0
        self.disconnect_calls = 0
        self._battery = 87

    def send_rc(self, lr: int, fb: int, ud: int, yaw: int) -> None:
        self.rc_calls.append((lr, fb, ud, yaw))

    def takeoff(self) -> None:
        self.takeoff_calls += 1

    def land(self) -> None:
        self.land_calls += 1

    def get_battery(self) -> int:
        self.battery_calls += 1
        return self._battery

    def stop_stream(self) -> None:
        self.stop_stream_calls += 1

    def disconnect(self) -> None:
        self.disconnect_calls += 1


class FakeClock:
    """Controllable monotonic clock for deterministic tests."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_shutdown_lands_when_airborne() -> None:
    controller = FakeController()
    session = FlightSession(controller, auto_start_ticker=False)
    session.takeoff()
    assert session.is_airborne

    session.shutdown_safe()

    assert controller.land_calls == 1
    assert controller.stop_stream_calls == 1
    assert controller.disconnect_calls == 1
    assert not session.is_airborne


def test_shutdown_skips_land_when_grounded() -> None:
    controller = FakeController()
    session = FlightSession(controller, auto_start_ticker=False)

    session.shutdown_safe()

    assert controller.land_calls == 0
    assert controller.stop_stream_calls == 1
    assert controller.disconnect_calls == 1


def test_rc_tick_sends_stick_while_set() -> None:
    controller = FakeController()
    clock = FakeClock()
    session = FlightSession(
        controller,
        auto_start_ticker=False,
        clock=clock,
        stick_timeout_s=0.25,
    )
    session.takeoff()
    session.apply_key_action(KeyAction.MOVE_FORWARD)

    session.tick()

    assert controller.rc_calls[-1] == (0, 40, 0, 0)


def test_stick_times_out_to_zero() -> None:
    controller = FakeController()
    clock = FakeClock()
    session = FlightSession(
        controller,
        auto_start_ticker=False,
        clock=clock,
        stick_timeout_s=0.25,
    )
    session.takeoff()
    session.apply_key_action(KeyAction.MOVE_LEFT)
    session.tick()
    assert controller.rc_calls[-1][0] == -40

    clock.advance(0.3)
    session.tick()

    assert controller.rc_calls[-1] == (0, 0, 0, 0)


def test_keepalive_while_airborne_idle() -> None:
    controller = FakeController()
    clock = FakeClock()
    session = FlightSession(
        controller,
        auto_start_ticker=False,
        clock=clock,
        keepalive_s=5.0,
    )
    session.takeoff()
    session.tick()  # initial zero RC after takeoff path may not send
    controller.rc_calls.clear()

    clock.advance(5.1)
    session.tick()

    assert (0, 0, 0, 0) in controller.rc_calls


def test_battery_is_cached() -> None:
    controller = FakeController()
    clock = FakeClock()
    session = FlightSession(
        controller,
        auto_start_ticker=False,
        clock=clock,
        battery_cache_s=1.0,
    )

    assert session.battery == 87
    assert session.battery == 87
    assert controller.battery_calls == 1

    clock.advance(1.1)
    assert session.battery == 87
    assert controller.battery_calls == 2


def test_clamp_distance_and_degrees() -> None:
    from vert_tracker.drone.controller import clamp_degrees, clamp_distance_cm

    assert clamp_distance_cm(10) == 20
    assert clamp_distance_cm(999) == 500
    assert clamp_degrees(0) == 1
    assert clamp_degrees(400) == 360

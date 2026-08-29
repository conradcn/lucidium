"""A fresh Session must declare its whole task surface up front.

Every one of these attributes used to be grafted onto the instance at
runtime by handlers.py / render_scheduler.py, which meant ``aclose`` had
no reliable inventory of what to cancel and a typo in an attribute name
silently orphaned a whole task group. Pin the contract here.
"""

from lucidium.orchestration.session import Session

# name -> expected initial value on a freshly constructed Session
_TASK_ATTRS = {
    "_foreground_task": None,
    "_foreground_stream_task": None,
    "_world_init_task": None,
    "_char_desc_task": None,
    "_name_options_task": None,
    "_preview_bg_task": None,
    "_preview_guide_task": None,
    "_pc_portrait_task": None,
    "_speculative_tasks": {},
    "_summarizer_tasks": [],
    "_music_tasks": [],
    "_asset_tasks": [],
}


def test_fresh_session_declares_every_task_attribute(tmp_path):
    session = Session(saves_root=tmp_path)
    for name, expected in _TASK_ATTRS.items():
        assert hasattr(session, name), f"Session is missing {name}"
        assert getattr(session, name) == expected, f"{name} should start as {expected!r}"


def test_owned_tasks_is_empty_on_a_fresh_session(tmp_path):
    session = Session(saves_root=tmp_path)
    assert session.owned_tasks() == []

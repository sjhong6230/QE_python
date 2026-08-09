"""Compatibility helpers for qepy-pw and upstream QE XML saves."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from ..pw.save import QES_NAMESPACE


_NS = {"qes": QES_NAMESPACE}


def upstream_qe_xml(root: ET.Element) -> bool:
    """Return whether only the upstream QE root carries the QES namespace."""
    return root.find("output") is not None and root.find("qes:output", _NS) is None


def find(root: ET.Element, path: str) -> ET.Element | None:
    """Find a slash path in either of QE's namespace serializations."""
    namespaced = "/".join(f"qes:{part}" for part in path.split("/"))
    element = root.find(namespaced, _NS)
    return element if element is not None else root.find(path)


def findall(root: ET.Element, path: str) -> list[ET.Element]:
    """Find all matching elements in either namespace serialization."""
    namespaced = "/".join(f"qes:{part}" for part in path.split("/"))
    entries = root.findall(namespaced, _NS)
    return entries if entries else root.findall(path)


def findtext(
    root: ET.Element, path: str, default: str | None = None
) -> str | None:
    """Read text from either namespace serialization."""
    element = find(root, path)
    return default if element is None or element.text is None else element.text


__all__ = ["find", "findall", "findtext", "upstream_qe_xml"]

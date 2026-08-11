from __future__ import annotations

import xml.etree.ElementTree as ET

from qepy_pw.pp.xml_data import find, findall, findtext, upstream_qe_xml


def test_helpers_read_fully_namespaced_qepy_xml() -> None:
    root = ET.fromstring(
        """<qes:espresso xmlns:qes='http://www.quantum-espresso.org/ns/qes/qes-1.0'>
        <qes:output><qes:band_structure>
          <qes:ks_energies><qes:eigenvalues>1 2</qes:eigenvalues></qes:ks_energies>
          <qes:ks_energies><qes:eigenvalues>3 4</qes:eigenvalues></qes:ks_energies>
        </qes:band_structure></qes:output>
        </qes:espresso>"""
    )

    records = findall(root, "output/band_structure/ks_energies")
    assert not upstream_qe_xml(root)
    assert find(root, "output/band_structure") is not None
    assert len(records) == 2
    assert findtext(records[1], "eigenvalues") == "3 4"
    assert findtext(root, "output/missing", "fallback") == "fallback"


def test_helpers_read_upstream_root_only_namespace_layout() -> None:
    root = ET.fromstring(
        """<qes:espresso xmlns:qes='http://www.quantum-espresso.org/ns/qes/qes-1.0'>
        <output><band_structure>
          <ks_energies><eigenvalues>1 2</eigenvalues></ks_energies>
        </band_structure></output>
        </qes:espresso>"""
    )

    assert upstream_qe_xml(root)
    assert findtext(root, "output/band_structure/ks_energies/eigenvalues") == "1 2"
    assert len(findall(root, "output/band_structure/ks_energies")) == 1

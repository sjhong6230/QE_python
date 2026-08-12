# Copyright (C) 2026 Seung-Ju Hong
# SPDX-License-Identifier: GPL-2.0-or-later
"""QE 7.5 namelist variable names used to classify input diagnostics.

This is not a second input parser.  It lets the scalar port distinguish a
valid-but-unported QE option from a misspelled/non-QE option.  Names are from
``Modules/input_parameters.f90`` in Quantum ESPRESSO 7.5.
"""

from __future__ import annotations


def _names(payload: str) -> frozenset[str]:
    return frozenset(payload.lower().split())


QE_NAMELIST_VARIABLES = {
    "control": _names(
        """
        title calculation verbosity restart_mode nstep iprint isave tstress
        tprnfor dt ndr ndw outdir prefix wfcdir max_seconds ekin_conv_thr
        etot_conv_thr forc_conv_thr pseudo_dir disk_io tefield dipfield lberry
        gdir nppstr wf_collect lelfield nberrycyc refg tefield2 saverho tabps
        use_wannier lecrpa lfcp tqmmm vdw_table_name lorbm memory
        point_label_type input_xml_schema_file gate trism twochem use_spinflip
        symmetry_with_labels
        """
    ),
    "system": _names(
        """
        ibrav celldm a b c cosab cosac cosbc nat ntyp nbnd ecutwfc ecutrho
        nr1 nr2 nr3 nr1s nr2s nr3s nr1b nr2b nr3b nosym nosym_evc noinv
        use_all_frac force_symmorphic starting_charge starting_magnetization
        occupations degauss nspin ecfixed qcutz q2sigma degauss_cond nbnd_cond
        nelec_cond lda_plus_u lda_plus_u_kind u_projection_type
        hubbard_parameters hubbard_u hubbard_j0 hubbard_j hubbard_v
        hubbard_u_back hubbard_alpha hubbard_alpha_back hubbard_beta hubbard_occ
        hub_pot_fix orbital_resolved reserv reserv_back dmft dmft_prefix edir
        emaxpos eopreg eamp smearing starting_ns_eigenvalue input_dft la2f
        assume_isolated nqx1 nqx2 nqx3 ecutfock localization_thr scdm ace
        scdmden scdmgrd nscdm n_proj exxdiv_treatment x_gamma_extrapolation
        yukawa ecutvcut exx_fraction screening_parameter ref_alat noncolin
        lspinorb starting_spin_angle lambda angle1 angle2 report lforcet
        constrained_magnetization b_field fixed_magnetization sic sic_epsilon
        force_pairing sic_alpha pol_type sic_gamma sic_energy sci_vb sci_cb
        tot_charge tot_magnetization one_atom_occupations vdw_corr london
        london_s6 london_rcut london_c6 london_rvdw dftd3_version
        dftd3_threebody ts_vdw ts_vdw_isolated ts_vdw_econv_thr xdm xdm_a1
        xdm_a2 mbd_vdw step_pen a_pen sigma_pen alpha_pen no_t_rev esm_bc
        esm_efield esm_w esm_nfit esm_debug esm_debug_gpmax esm_a esm_zb lgcscf
        gcscf_ignore_mun gcscf_mu gcscf_conv_thr gcscf_gk gcscf_gh gcscf_beta
        space_group uniqueb origin_choice rhombohedral zgate relaxz block
        block_1 block_2 block_height nextffield
        """
    ),
    "electrons": _names(
        """
        emass emass_cutoff orthogonalization exx_maxstep electron_maxstep
        scf_must_converge ortho_eps ortho_max electron_dynamics
        electron_damping electron_velocities electron_temperature ekincw fnosee
        ampre grease diis_size diis_nreset diis_hcut diis_wthr diis_delt
        diis_maxstep diis_rot diis_fthr diis_temp diis_achmix diis_g0chmix
        diis_g1chmix diis_nchmix diis_nrot diis_rothr diis_ethr diis_chguess
        mixing_mode mixing_beta mixing_ndim mixing_fixed_ns tqr tq_smoothing
        tbeta_smoothing diago_cg_maxiter diago_david_ndim diago_rmm_ndim
        diago_rmm_conv diago_gs_nblock diagonalization startingpot startingwfc
        conv_thr adaptive_thr conv_thr_init conv_thr_multi diago_thr_init
        n_inner fermi_energy rotmass occmass rotation_damping occupation_damping
        rotation_dynamics occupation_dynamics tcg maxiter etresh passop epol
        efield epol2 efield2 diago_full_acc occupation_constraints
        niter_cg_restart niter_cold_restart lambda_cold efield_cart real_space
        tcpbo emass_emin emass_cutoff_emin electron_damping_emin dt_emin
        efield_phase pre_state
        """
    ),
}

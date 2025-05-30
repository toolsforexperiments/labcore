try:
    from qm.QuantumMachinesManager import QuantumMachinesManager
except:
    from qm.quantum_machines_manager import QuantumMachinesManager


def close_my_qm(config, host, port):
    """
    Helper function that closes any machines that is open in the OPT with host and port that uses any controller that
    is present in the passed config.

    Parameters
    ----------
    config
        Config dictionary from which we are trying to open a QuantumMachine
    host
        The OPT host ip address
    port
        The OPT port

    """
    qmm = QuantumMachinesManager(host=host, port=port)
    controllers = [con for con in config['controllers'].keys()]
    open_qms = [qmm.get_qm(machine_id=machine_id) for machine_id in qmm.list_open_quantum_machines()]
    for qm in open_qms:
        for con in controllers:
            if con in qm.list_controllers():
                qm.close()

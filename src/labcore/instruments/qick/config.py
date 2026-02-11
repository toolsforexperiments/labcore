import logging

import Pyro4
from qick import QickConfig


logger = logging.getLogger(__name__)

class QBoardConfig:


    def __init__(self, params, nameserver_host="192.168.1.1", nameserver_port=8888, nameserver_name="rfsoc"):
        """
        
        Check the reference, https://qick-docs.readthedocs.io/en/latest/_autosummary/qick.qick_asm.html#qick.qick_asm.QickConfig 
         for some useful conversion function that exist


        :param params: Proxy instance of the parameter manager to to get values.
        :param nameserver_host: IP of the host nameserver. This is usually the measurement PC you are using.
        :param nameserver_port: Port of the host nameserver.
        :param nameserver_name: Name of the nameserver. This needs to match between the qick and measurement computer

        """
        self.params = params
        self.nameserver_host = nameserver_host
        self.nameserver_port = nameserver_port
        self.nameserver_name = nameserver_name
        self.soc = None
        self.soccfg = None # Network configuration


    def generate_soccfg(self) -> QickConfig:
        """
        Generates the network configuration from the nameserver. Sotres the soc and soccfg in the class.

        :return: soccfg: QickConfig instance.
        """

        Pyro4.config.SERIALIZER = 'pickle'
        Pyro4.config.PICKLE_PROTOCOL_VERSION = 4
        ns = Pyro4.locateNS(host=self.nameserver_host, port=self.nameserver_port)
        soc = Pyro4.Proxy(ns.lookup(self.nameserver_name))
        soccfg = QickConfig(soc.get_cfg())
        self.soc = soc
        self.soccfg = soccfg
        logger.info("Generated soccfg")
        logger.info(soccfg)
        return soccfg


    # TODO: See if there is a way of checking if there has to be some way of checking for the required parameters like reps and expts that some qick classes require.
    def config(self):
        """
        Generates the configuration and the updates configuration of the qick.

        Returns both the network configuration and the qick configuration. 
        Both are needed to start a measurement.
        """
        if self.soc is None or self.soccfg is None:
            self.generate_soccfg()

        conf = self.config_()
        # If you are using the averager program this needs to be a part of the config
        if "reps" not in conf:
            try:
                conf["reps"] = self.params.reps()
            except Exception as e:
                raise AttributeError("Could not get reps from parameter manager, please provide reps in the configuration or parameter manager")


        return self.soccfg, conf

    def config_(self):
        raise NotImplementedError("config_() method must be implemented in subclass")
# Fridge Dashboard


The fridge dashboard is designed to have an easy way to view the status and history of temperatures, pressures, and other parameters taken from the fridge computers. It also has capabilities for sending alerts to different messaging services, such as Slack based on parameter status.

!!! Note
    The following portion assumes the user has:
    - instrumentserver installed and has basic familiarity with it and using config files
    - labcore installed


## The Instrumentserver


### Config File

In order to use the dashboard, we will need to have an instance of the instrumentserver running to fetch data from the fridge computer. Below is an example configuration file that can be used for the dashboard.


```yaml
instruments:
  
  fridge_nh:
    type: labcore.instruments.qcodes_drivers.Oxford.triton.OxfordTriton
    address: 192.168.1.24
    init: 
      port: 33576
      temp_channel_mapping:
          T1: "PT2_HEAD"
          T2: "PT2_PLATE"
          T3: "STILL_PLATE"
          T4: "COLD_PLATE"
          T6: "PT1_HEAD"
          T7: "PT1_PLATE"
          T8: "MC_PLATE_RUO2"

    pollingRate:
      comp_state: 15
      PT1_PLATE: 15
      PT2_PLATE: 15
      STILL_PLATE: 15
      COLD_PLATE: 15
      MC_PLATE_RUO2: 15
      turb1_state: 15
      turb1_speed: 15

networking:
  externalBroadcast: "tcp://128.174.165.137:6000"
```


As usual, we declare an instrument for the fridge. For the driver, use the following from labcore if using an Oxford Fridge:

```
labcore.instruments.qcodes_drivers.Oxford.triton.OxfordTriton
```

For `address`, fill out the field with the IPv4 address of the fridge computer (on the same network as the computer you are running the instrumentserver on)

Then, in the `init` field, there are two items that must be filled out:

From the manual of the fridge being used, provide the port on the fridge computer to communicate with. (For Oxford Triton, this port is `33576`). 

Then, create a dictionary containing the mapping between name and temperature channel for each channel you would like a named parameter for. In the provided config, 7 Temperature Channels are used to create named parameters. Channels can be found on the Lakeshore thermometry dialog on the fridge computer.

Then, under the field `pollingRate`, provide a dictionary for how often to poll each parameter given. The dictionary is constructed with the name of the parameter, and the interval to poll (in seconds)

Finally, fill out the field `networking`. For the dashboard, only one field is required, `externalBroadcast`. Locate the IPv4 address of the computer you are running the instrumentserver on for the internet network you wish to broadcast the data to. Include the port to broadcast to as well.


### Starting the Instrument Server


Use the following to start the instrumentserver. Replace serverConfig.yml with the path to the config file you created above.

```bash
$ instrumentserver -c serverConfig.yml --gui False
```
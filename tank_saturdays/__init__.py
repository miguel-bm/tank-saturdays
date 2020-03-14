from gym.envs.registration import register

register(
	id='TankSaturdays-v0',
	entry_point='tank_saturdays.envs:TankSaturdays',
)

#ifndef STM32F411X_I2C_H_
#define STM32F411X_I2C_H_

#include "STM32F411x_defs.h"

#include <stdint.h>

#define ST_I2C_ACK_DISABLE 		0
#define ST_I2C_ACK_ENABLE  		1

#define ST_I2C_ADDRMODE_7_BIT	0

#define ST_I2C_MODE_STD 		(100U * 1000U)
#define ST_I2C_MODE_FAST   		(400U * 1000U)

/// Duty cycle for the I2C "fast" mode, T_low = 2 * T_high
#define ST_I2C_FM_DUTY_CYCLE_2		0
/// Duty cycle for the "fast" mode, T_low = (16/9) * T_high
#define ST_I2C_FM_DUTY_CYCLE_16_9	1

typedef struct
{
	uint32_t mode;			// ST_I2C_MODE_*
	uint8_t ack_enable;		// ST_I2C_ACK_*
	uint8_t slave_address;	// 7 bits of slave address
	uint8_t fm_duty_cycle;	// ST_I2C_FM_DUTY_CYCLE_*
} ST_I2C_conf_t;

typedef struct
{
	ST_I2C_reg_t * baseAdress;
	ST_I2C_conf_t config;
} ST_I2C_t;

extern void ST_I2C_init(ST_I2C_t *i2c);
extern void ST_I2C_deinit(ST_I2C_t *i2c);

extern void ST_I2C_Master_send(ST_I2C_t *i2c, const uint8_t slave_addr, const uint8_t *tx_buffer, const size_t data_len);

#endif // STM32F411X_I2C_H_

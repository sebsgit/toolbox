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

#define ST_I2C_IRQ_STATE_IDLE 		(0)
#define ST_I2C_IRQ_STATE_BUSY_RX	(1)
#define ST_I2C_IRQ_STATE_BUSY_TX	(2)

typedef struct
{
	uint32_t mode;			// ST_I2C_MODE_*
	uint8_t ack_enable;		// ST_I2C_ACK_*
	uint8_t slave_address;	// 7 bits of slave address
	uint8_t fm_duty_cycle;	// ST_I2C_FM_DUTY_CYCLE_*
} ST_I2C_conf_t;

typedef struct
{
	const uint8_t *tx_buffer;
	uint8_t *rx_buffer;
	uint32_t tx_buff_len;
	uint32_t rx_buff_len;
	uint8_t irq_state;		// ST_I2C_IRQ_STATE_*
	uint8_t dev_addr;
	uint32_t rx_size;
	uint8_t rep_start;
} ST_I2C_irq_data_t;

typedef struct
{
	ST_I2C_reg_t * baseAdress;
	ST_I2C_conf_t config;
	ST_I2C_irq_data_t irq;
} ST_I2C_t;

extern void ST_I2C_init(ST_I2C_t *i2c);
extern void ST_I2C_deinit(ST_I2C_t *i2c);

// blocking API
extern void ST_I2C_Master_send(ST_I2C_t *i2c, const uint8_t slave_addr, const uint8_t *tx_buffer, const size_t data_len);

// blocking API
extern void ST_I2C_Master_receive(ST_I2C_t *i2c, const uint8_t slave_addr, uint8_t *rx_buffer, const size_t data_len);

extern void ST_I2C_irq_control(ST_I2C_t *i2c, uint8_t priority, uint8_t enable);

// interrupt-based API
extern uint8_t ST_I2C_Master_send_IT(ST_I2C_t *i2c, const uint8_t slave_addr, const uint8_t *tx_buffer, const size_t data_len);

// interrupt-based API
extern uint8_t ST_I2C_Master_receive_IT(ST_I2C_t *i2c, const uint8_t slave_addr, uint8_t *rx_buffer, const size_t data_len);


#endif // STM32F411X_I2C_H_

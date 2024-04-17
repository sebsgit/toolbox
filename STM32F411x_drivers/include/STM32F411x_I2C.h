#ifndef STM32F411X_I2C_H_
#define STM32F411X_I2C_H_

#include "STM32F411x_defs.h"

#include <stdint.h>

// I2C register bit positions
#define ST_I2C_SR1_SB			(0)
#define ST_I2C_SR1_ADDR			(1)
#define ST_I2C_SR1_BTF			(2)
#define ST_I2C_SR1_ADDR10		(3)
#define ST_I2C_SR1_STOPF		(4)
#define ST_I2C_SR1_RXNE			(6)
#define ST_I2C_SR1_TXE			(7)
#define ST_I2C_SR1_BERR			(8)		// bus error
#define ST_I2C_SR1_ARLO			(9)		// arbitration loss
#define ST_I2C_SR1_AF			(10)	// ACK failure
#define ST_I2C_SR1_OVR			(11)	// over- or underrun
#define ST_I2C_SR1_PEC			(12)	// PEC error
#define ST_I2C_SR1_TMOUT		(14)	// timeout error

#define ST_I2C_SR2_MSL			(0)		// master or slave
#define ST_I2C_SR2_TRA			(2)		// transmitter or receiver

#define ST_I2C_CR2_ITERREN		(8)
#define ST_I2C_CR2_ITEVEN		(9)
#define ST_I2C_CR2_ITBUFEN		(10)

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

#define ST_I2C_EVENT_RX_COMPL		(1)
#define ST_I2C_EVENT_TX_COMPL		(2)
#define ST_I2C_EVENT_STOP			(3)
#define ST_I2C_EVENT_ERR_BUS		(4)
#define ST_I2C_EVENT_ERR_ARBITRATION_LOSS (5)
#define ST_I2C_EVENT_ERR_ACK_FAIL	(6)
#define ST_I2C_EVENT_ERR_OVERRUN	(7)
#define ST_I2C_EVENT_ERR_PEC		(8)
#define ST_I2C_EVENT_ERR_TIMEOUT	(9)
#define ST_I2C_EVENT_SLAVE_TRANSMIT	(10)
#define ST_I2C_EVENT_SLAVE_RECEIVE	(11)

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

extern uint8_t ST_I2C_is_master(ST_I2C_t *i2c);

// blocking API
extern void ST_I2C_Master_send(ST_I2C_t *i2c, const uint8_t slave_addr, const uint8_t *tx_buffer, const size_t data_len);

// blocking API
extern void ST_I2C_Master_receive(ST_I2C_t *i2c, const uint8_t slave_addr, uint8_t *rx_buffer, const size_t data_len);

extern void ST_I2C_data_write(ST_I2C_t *i2c, const uint8_t data);
extern uint8_t ST_I2C_data_read(ST_I2C_t *i2c);

extern void ST_I2C_irq_control(ST_I2C_t *i2c, uint8_t priority, uint8_t enable);
extern void ST_I2C_callback_control(ST_I2C_t *i2c, uint8_t enable);

extern void ST_I2C_irq_ev_handler(ST_I2C_t *i2c);
extern void ST_I2C_irq_err_handler(ST_I2C_t *i2c);

// interrupt-based API
extern uint8_t ST_I2C_Master_send_IT(ST_I2C_t *i2c, const uint8_t slave_addr, const uint8_t *tx_buffer, const size_t data_len);

// interrupt-based API
extern uint8_t ST_I2C_Master_receive_IT(ST_I2C_t *i2c, const uint8_t slave_addr, uint8_t *rx_buffer, const size_t data_len);

extern void ST_I2C_App_Event(ST_I2C_t *pSpi, uint8_t e_type) __attribute__((weak));

#endif // STM32F411X_I2C_H_

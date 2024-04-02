#ifndef STM32F411X_SPI_H_
#define STM32F411X_SPI_H_

#include "STM32F411x_defs.h"

#include <stdint.h>

typedef struct
{
	uint8_t mode;				// master or slave (ST_SPI_MASTER/SLAVE)
	uint8_t bus_config;			// transfer mode selection (ST_SPI_MODE_*)
	uint8_t clock_speed;		// clock speed = (peripheral clock) / (ST_SPI_CLOCK_PRESCALER_*)
	uint8_t data_frame_format;	// 8 or 16 bit data frame (ST_SPI_DFF_*)
	uint8_t clock_polarity;		// value on clock idle (ST_SPI_CLOCK_IDLE_*)
	uint8_t clock_phase;		// when to capture data (ST_SPI_CLOCK_PHASE_EDGE_*)
	uint8_t ssm;				// software or hardware "nss" pin management (ST_SPI_SW_SLAVE_*)
	uint8_t ti_enable;			// TI or Motorola transfer mode (ST_SPI_TI_MODE_*)
} ST_SPI_conf_t;

typedef struct
{
	const uint8_t *tx_buff;
	uint32_t tx_buff_len;
	uint8_t *rx_buff;
	uint32_t rx_buff_len;
	uint8_t tx_state;		// status of the transmission (ST_SPI_IRQ_*)
	uint8_t rx_state;		// status of the transmission (ST_SPI_IRQ_*)
} ST_SPI_irq_data_t;

typedef struct
{
	ST_SPI_reg_t *baseAddress;
	ST_SPI_conf_t config;
	ST_SPI_irq_data_t irq;
} ST_SPI_t;

#define ST_SPI_SLAVE				(0)
#define ST_SPI_MASTER				(1)
#define ST_SPI_MODE_FULL_DUPLEX		(0)
#define ST_SPI_MODE_HALF_DUPLEX		(1)
#define ST_SPI_MODE_SIMPLEX_TX		(2)
#define ST_SPI_MODE_SIMPLEX_RX		(3)
#define ST_SPI_USE_2_LINES			(0)
#define ST_SPI_USE_1_LINE			(1)
#define ST_SPI_OUTPUT_DI			(0)
#define ST_SPI_OUTPUT_EN			(1)
#define ST_SPI_RX_ONLY_OFF			(0)
#define ST_SPI_RX_ONLY_ON			(1)
#define ST_SPI_DFF_8Bit				(0)
#define ST_SPI_DFF_16bit			(1)
#define ST_SPI_SW_SLAVE_OFF			(0)
#define ST_SPI_SW_SLAVE_ON			(1)
#define ST_SPI_CLOCK_PRESCALER_2	(0x0)
#define ST_SPI_CLOCK_PRESCALER_4	(0x1)
#define ST_SPI_CLOCK_PRESCALER_8	(0x2)
#define ST_SPI_CLOCK_PRESCALER_16	(0x3)
#define ST_SPI_CLOCK_PRESCALER_32	(0x4)
#define ST_SPI_CLOCK_PRESCALER_64	(0x5)
#define ST_SPI_CLOCK_PRESCALER_128	(0x6)
#define ST_SPI_CLOCK_PRESCALER_256	(0x7)
#define ST_SPI_CLOCK_IDLE_0			(0)
#define ST_SPI_CLOCK_IDLE_1			(1)
#define ST_SPI_CLOCK_PHASE_EDGE_1	(0)
#define ST_SPI_CLOCK_PHASE_EDGE_2	(1)
#define ST_SPI_TI_MODE_ON			(1)
#define ST_SPI_TI_MODE_OFF			(0)
#define ST_SPI_IRQ_IDLE				(0)
#define ST_SPI_IRQ_BUSY_RX			(1)
#define ST_SPI_IRQ_BUSY_TX			(2)

#define ST_SPI_EVENT_RX_COMPL		(1)
#define ST_SPI_EVENT_TX_COMPL		(2)

extern void ST_SPI_clock_control(ST_SPI_reg_t *pSpiReg, uint8_t enable);

extern void ST_SPI_init(ST_SPI_t * pSpi);
extern void ST_SPI_deinit(ST_SPI_reg_t * pSpiReg);

/// blocking call
extern void ST_SPI_send(ST_SPI_reg_t* pSpiReg, const uint8_t* data, const size_t data_len);
// blocking call
extern void ST_SPI_recv(ST_SPI_reg_t* pSpiReg, uint8_t* data, const size_t data_len);

extern void ST_SPI_IRQ_control(ST_SPI_reg_t * pSpiReg, uint8_t priority, uint8_t enable);

// non-blocking call
extern uint8_t ST_SPI_send_irq(ST_SPI_t *pSpi, const uint8_t* data, const size_t data_len);
// non-blocking call
extern uint8_t ST_SPI_recv_irq(ST_SPI_t *pSpi, uint8_t* data, const size_t data_len);

extern void ST_SPI_IRQ_handle(ST_SPI_t *pSpi);

extern void ST_SPI_App_Event(ST_SPI_t *pSpi, uint8_t e_type) __attribute__((weak));

#endif // STM32F411X_SPI_H_

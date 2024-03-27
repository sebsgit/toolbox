#ifndef STM32F411X_SPI_H_
#define STM32F411X_SPI_H_

#include "STM32F411x_defs.h"

#include <stdint.h>

typedef struct
{
	uint8_t mode;				// master or slave (ST_SPI_MASTER/SLAVE)
	uint8_t bus_config;
	uint8_t clock_speed;		// clock speed (ST_SPI_CLOCK_PRESCALER_*)
	uint8_t data_frame_format;	// 8 or 16 bit data frame (ST_SPI_DFF_*)
	uint8_t clock_polarity;		// value on clock idle (ST_SPI_CLOCK_IDLE_*)
	uint8_t clock_phase;		// when to capture data (ST_SPI_CLOCK_PHASE_EDGE_*)
	uint8_t ssm;				// software or hardware "nss" pin management (ST_SPI_SW_SLAVE_*)
} ST_SPI_conf_t;

typedef struct
{
	ST_SPI_reg_t *baseAddress;
	ST_SPI_conf_t config;
} ST_SPI_t;

#define ST_SPI_DFF_8Bit				(0)
#define ST_SPI_DFF_16bit			(1)
#define ST_SPI_SW_SLAVE_OFF			(0)
#define ST_SPI_SW_SLAVE_ON			(1)
#define ST_SPI_SLAVE				(0)
#define ST_SPI_MASTER				(1)
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

extern void ST_SPI_clock_control(ST_SPI_reg_t *pSpiReg, uint8_t enable);

extern void ST_SPI_init(ST_SPI_t * pSpi);
extern void ST_SPI_deinit(ST_SPI_reg_t * pSpiReg);

extern void ST_SPI_send(ST_SPI_reg_t* pSpiReg, const uint8_t* data, const size_t data_len);
extern void ST_SPI_recv(ST_SPI_reg_t* pSpiReg, uint8_t* data, const size_t data_len);

#endif // STM32F411X_SPI_H_

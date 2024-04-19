#ifndef STM32F411X_UART_H_
#define STM32F411X_UART_H_

#include "STM32F411x_defs.h"

#include <stdint.h>

#define ST_UART_MODE_RX		(0)
#define ST_UART_MODE_TX		(1)
#define ST_UART_MODE_RXTX	(2)

#define ST_UART_BAUD_9600	(9600)
#define ST_UART_BAUD_115200	(115200)

#define ST_UART_WORD_8		(0)
#define ST_UART_WORD_9		(1)

#define ST_UART_STOPB_1		(0)
#define ST_UART_STOPB_2		(1)

#define ST_UART_PARITY_ODD	(1)
#define ST_UART_PARITY_EVEN (0)
#define ST_UART_PARITY_OFF	(2)

#define ST_UART_HWFLOW_OFF	(0)
#define ST_UART_HWFLOW_CTS	(1)
#define ST_UART_HWFLOW_RTS	(1)
#define ST_UART_HWFLOW_ALL	(1)

#define ST_UART_OVERSAMPLE_8	(1)
#define ST_UART_OVERSAMPLE_16	(0)

#define ST_UART_CR1_OVERS	(15)
#define ST_UART_CR1_EN		(13)
#define ST_UART_CR1_WLEN	(12)
#define ST_UART_CR1_PARCTL	(10)
#define ST_UART_CR1_PARSEL	(9)
#define ST_UART_CR1_PEIRQ	(8)
#define ST_UART_CR1_TXEIRQ	(7)
#define ST_UART_CR1_TCEIRQ	(6)
#define ST_UART_CR1_RXNEIRQ	(5)
#define ST_UART_CR1_IDLEIRQ	(4)
#define ST_UART_CR1_TXEN	(3)
#define ST_UART_CR1_RXEN	(2)

#define ST_UART_CR2_STOPH	(13)
#define ST_UART_CR2_STOPL	(12)

#define ST_UART_CR3_CTS		(9)
#define ST_UART_CR3_RTS		(8)

#define ST_UART_SR_TXE		(7)
#define ST_UART_SR_TC		(6)
#define ST_UART_SR_RXNE		(5)

typedef struct
{
	uint8_t mode;			// ST_UART_MODE_*
	uint32_t baudRate;		// ST_UART_BAUD_*
	uint8_t wordLength;		// ST_UART_WORD_*
	uint8_t stopBits;		// ST_UART_STOPB_*
	uint8_t parity;			// ST_UART_PARITY_*
	uint8_t hwFlow;			// ST_UART_HWFLOW_*
	uint8_t oversample;		// ST_UART_OVERSAMPLE_*
} ST_UART_conf_t;

typedef struct
{
	ST_UART_conf_t conf;
	ST_UART_reg_t *baseAddress;
} ST_UART_t;

extern void ST_UART_init(ST_UART_t *uart);
extern void ST_UART_deinit(ST_UART_reg_t *uart);

extern void ST_UART_irq_config(ST_UART_t *uart, uint8_t priority, uint8_t enable);

extern void ST_UART_write(ST_UART_t *uart, const uint8_t *data, const uint32_t data_len);

#endif // STM32F411X_UART_H_

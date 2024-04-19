#include "STM32F411x_UART.h"

static void ST_UART_enable_clock(ST_UART_t *uart)
{
	switch ((uint32_t)uart->baseAddress)
	{
	case ST_USART1_BASE_ADDRESS:
		ST_USART1_CLOCK_EN();
		break;
	case ST_USART2_BASE_ADDRESS:
		ST_USART2_CLOCK_EN();
		break;
	case ST_USART6_BASE_ADDRESS:
		ST_USART6_CLOCK_EN();
		break;
	default:
		break;
	}
}

static uint32_t ST_UART_get_irq_no(ST_UART_t *uart)
{
	switch ((uint32_t)uart->baseAddress)
	{
	case ST_USART1_BASE_ADDRESS:
		return ST_NVIC_IRQ_USART1;
	case ST_USART2_BASE_ADDRESS:
		return ST_NVIC_IRQ_USART2;
	case ST_USART6_BASE_ADDRESS:
		return ST_NVIC_IRQ_USART6;
	default:
		return 0;
	}
}

static uint32_t ST_UART_get_clock_freq(ST_UART_reg_t *uart)
{
	if (uart == ST_USART2)
	{
		return ST_RCC_get_apb1_clock_frequency();
	}
	else
	{
		return ST_RCC_get_apb2_clock_frequency();
	}
}

static void ST_UART_set_baud_rate(ST_UART_t *uart)
{
	const uint32_t clock_freq = ST_UART_get_clock_freq(uart->baseAddress);
	uint32_t brr = 0;

	// from the datasheet: BAUD = (F_clck) / (8 * (oversampling) * USARTDIV)
	// where, oversampling = 1: over8, 2: over16
	// USARTDIV should be programmed into the BRR register (=> USARTDIV = (F_ck) / (8 * oversampling * BAUD))

	const uint32_t oversampling = (uart->conf.oversample == ST_UART_OVERSAMPLE_16) ? 2 : 1;
	const uint32_t divider = (8 * oversampling * uart->conf.baudRate);
	const uint32_t usartdiv = ((clock_freq * 100) / divider);
	const uint32_t mantissa = usartdiv / 100;

	uint32_t fraction = (usartdiv - (mantissa * 100));
	if (uart->conf.oversample == ST_UART_OVERSAMPLE_8)
	{
		fraction = (((fraction * 8) + 50) / 100) & ((uint8_t)0x07);
	}
	else
	{
		fraction = (((fraction * 16) + 50) / 100) & ((uint8_t)0x0F);
	}

	brr |= fraction;
	brr |= (mantissa << 4);

	uart->baseAddress->BRR = brr;
}

void ST_UART_init(ST_UART_t *uart)
{
	ST_UART_enable_clock(uart);
	if ((uart->conf.mode == ST_UART_MODE_RX) || (uart->conf.mode == ST_UART_MODE_RXTX))
	{
		uart->baseAddress->CR1 |= (1 << ST_UART_CR1_RXEN);
	}
	if ((uart->conf.mode == ST_UART_MODE_TX) || (uart->conf.mode == ST_UART_MODE_RXTX))
	{
		uart->baseAddress->CR1 |= (1 << ST_UART_CR1_TXEN);
	}
	if (uart->conf.oversample == ST_UART_OVERSAMPLE_8)
	{
		uart->baseAddress->CR1 |= (1 << ST_UART_CR1_OVERS);
	}
	else
	{
		uart->baseAddress->CR1 &= ~(1 << ST_UART_CR1_OVERS);
	}
	if (uart->conf.parity == ST_UART_PARITY_OFF)
	{
		uart->baseAddress->CR1 &= ~(1 << ST_UART_CR1_PARCTL);
	}
	else
	{
		uart->baseAddress->CR1 |= (1 << ST_UART_CR1_PARCTL);
		if (uart->conf.parity == ST_UART_PARITY_EVEN)
		{
			uart->baseAddress->CR1 &= ~(1 << ST_UART_CR1_PARSEL);
		}
		else
		{
			uart->baseAddress->CR1 |= (1 << ST_UART_CR1_PARSEL);
		}
	}
	if (uart->conf.wordLength == ST_UART_WORD_8)
	{
		uart->baseAddress->CR1 &= ~(1 << ST_UART_CR1_WLEN);
	}
	else
	{
		uart->baseAddress->CR1 |= (1 << ST_UART_CR1_WLEN);
	}
	if (uart->conf.stopBits == ST_UART_STOPB_1)
	{
		uart->baseAddress->CR2 &= ~(1 << ST_UART_CR2_STOPL);
		uart->baseAddress->CR2 &= ~(1 << ST_UART_CR2_STOPH);
	}
	else
	{
		uart->baseAddress->CR2 &= ~(1 << ST_UART_CR2_STOPL);
		uart->baseAddress->CR2 |= (1 << ST_UART_CR2_STOPH);
	}
	if ((uart->conf.hwFlow == ST_UART_HWFLOW_CTS) || (uart->conf.hwFlow == ST_UART_HWFLOW_ALL))
	{
		uart->baseAddress->CR3 |= (1 << ST_UART_CR3_CTS);
	}
	if ((uart->conf.hwFlow == ST_UART_HWFLOW_RTS) || (uart->conf.hwFlow == ST_UART_HWFLOW_ALL))
	{
		uart->baseAddress->CR3 |= (1 << ST_UART_CR3_RTS);
	}
	ST_UART_set_baud_rate(uart);

	uart->baseAddress->CR1 |= (1 << ST_UART_CR1_EN);
}

void ST_UART_deinit(ST_UART_reg_t *uart)
{
	switch ((uint32_t)uart)
	{
	case ST_USART1_BASE_ADDRESS:
		ST_RCC->APB2RST |= (1 << 4);
		break;
	case ST_USART2_BASE_ADDRESS:
		ST_RCC->APB1RST |= (1 << 17);
		break;
	case ST_USART6_BASE_ADDRESS:
		ST_RCC->APB2RST |= (1 << 5);
		break;
	default:
		break;
	}
}

void ST_UART_irq_config(ST_UART_t *uart, uint8_t priority, uint8_t enable)
{
	const uint32_t irq_no = ST_UART_get_irq_no(uart);
	if (irq_no == 0)
	{
		return;
	}
	ST_NVIC_configure_interrupt(irq_no, priority, enable);
}

void ST_UART_write(ST_UART_t *uart, const uint8_t *data, const uint32_t data_len)
{
	for (uint32_t i = 0; i < data_len; ++i)
	{
		// wait for the "TX empty" flag
		while (!(uart->baseAddress->SR & (1 << ST_UART_SR_TXE)));

		if (uart->conf.wordLength == ST_UART_WORD_8)
		{
			uart->baseAddress->DR = data[i];
		}
		else
		{
			//TODO 9 bit logic
		}
	}

	// wait for "transmission complete" flag
	while (!(uart->baseAddress->SR & (1 << ST_UART_SR_TC)));
}

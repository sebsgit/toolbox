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

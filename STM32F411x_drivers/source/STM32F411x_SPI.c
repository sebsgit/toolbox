#include "STM32F411x_SPI.h"

#define SET_BIT(where, which, value) do { if (value) { where |= (1 << (which)); } else { where &= ~(1 << (which)); } } while (0);

void ST_SPI_init(ST_SPI_t * pSpi)
{
	ST_SPI_clock_control(pSpi->baseAddress, 1);

	pSpi->baseAddress->CR1 &= ~(1 << 6); // disable SPI before applying new config

	SET_BIT(pSpi->baseAddress->CR2, 4, pSpi->config.ti_enable);
	if ((pSpi->config.mode == ST_SPI_MASTER) && !pSpi->config.ti_enable)
	{
		SET_BIT(pSpi->baseAddress->CR2, 2, 1); // SSOE (slave-select output enable)
	}

	// bus mode
	if (pSpi->config.bus_config == ST_SPI_MODE_FULL_DUPLEX)
	{
		SET_BIT(pSpi->baseAddress->CR1, 15, ST_SPI_USE_2_LINES);
		SET_BIT(pSpi->baseAddress->CR1, 14, ST_SPI_OUTPUT_EN);
	}
	else if (pSpi->config.bus_config == ST_SPI_MODE_HALF_DUPLEX)
	{
		SET_BIT(pSpi->baseAddress->CR1, 15, ST_SPI_USE_2_LINES);
		SET_BIT(pSpi->baseAddress->CR1, 14, ST_SPI_OUTPUT_EN);
	}
	else if (pSpi->config.bus_config == ST_SPI_MODE_SIMPLEX_RX)
	{
		SET_BIT(pSpi->baseAddress->CR1, 15, ST_SPI_USE_1_LINE);
		SET_BIT(pSpi->baseAddress->CR1, 14, ST_SPI_OUTPUT_DI);
		SET_BIT(pSpi->baseAddress->CR1, 10, ST_SPI_RX_ONLY_ON);
	}
	else if (pSpi->config.bus_config == ST_SPI_MODE_SIMPLEX_TX)
	{
		SET_BIT(pSpi->baseAddress->CR1, 15, ST_SPI_USE_1_LINE);
		SET_BIT(pSpi->baseAddress->CR1, 14, ST_SPI_OUTPUT_EN);
		SET_BIT(pSpi->baseAddress->CR1, 10, ST_SPI_RX_ONLY_OFF);
	}

	// master / slave select, bit 2
	SET_BIT(pSpi->baseAddress->CR1, 2, pSpi->config.mode);

	// ssm, bit 9
	SET_BIT(pSpi->baseAddress->CR1, 9, pSpi->config.ssm);

	// DFF, bit 11
	SET_BIT(pSpi->baseAddress->CR1, 11, pSpi->config.data_frame_format);

	// clock polarity, bit 1
	SET_BIT(pSpi->baseAddress->CR1, 1, pSpi->config.clock_polarity);

	// clock phase, bit 0
	SET_BIT(pSpi->baseAddress->CR1, 0, pSpi->config.clock_phase);

	// baud rate, bits 3-5
	SET_BIT(pSpi->baseAddress->CR1, 3, pSpi->config.clock_speed & 0x1);
	SET_BIT(pSpi->baseAddress->CR1, 4, pSpi->config.clock_speed & 0x2);
	SET_BIT(pSpi->baseAddress->CR1, 5, pSpi->config.clock_speed & 0x4);

	pSpi->baseAddress->CR1 |= (1 << 6);
}

void ST_SPI_deinit(ST_SPI_reg_t * pSpiReg)
{
	switch ((uint32_t)pSpiReg)
	{
		case ST_SPI1_BASE_ADDRESS:
			ST_RCC->APB2RST |= (1 << 12);
			break;
		case ST_SPI2_BASE_ADDRESS:
			ST_RCC->APB1RST |= (1 << 14);
			break;
		case ST_SPI3_BASE_ADDRESS:
			ST_RCC->APB1RST |= (1 << 15);
			break;
		case ST_SPI4_BASE_ADDRESS:
			ST_RCC->APB2RST |= (1 << 13);
			break;
		case ST_SPI5_BASE_ADDRESS:
			ST_RCC->APB2RST |= (1 << 20);
			break;
		default:
			break;
	}
}

void ST_SPI_clock_control(ST_SPI_reg_t *pSpiReg, uint8_t enable)
{
	switch ((uint32_t)pSpiReg)
	{
	case ST_SPI1_BASE_ADDRESS:
		if (enable)
		{
			ST_SPI1_CLOCK_EN();
		}
		else
		{
			ST_SPI1_CLOCK_DI();
		}
		break;
	case ST_SPI2_BASE_ADDRESS:
		if (enable)
		{
			ST_SPI2_CLOCK_EN();
		}
		else
		{
			ST_SPI2_CLOCK_DI();
		}
		break;
	case ST_SPI3_BASE_ADDRESS:
		if (enable)
		{
			ST_SPI3_CLOCK_EN();
		}
		else
		{
			ST_SPI3_CLOCK_DI();
		}
		break;
	case ST_SPI4_BASE_ADDRESS:
		if (enable)
		{
			ST_SPI4_CLOCK_EN();
		}
		else
		{
			ST_SPI4_CLOCK_DI();
		}
		break;
	case ST_SPI5_BASE_ADDRESS:
		if (enable)
		{
			ST_SPI5_CLOCK_EN();
		}
		else
		{
			ST_SPI5_CLOCK_DI();
		}
		break;
	default:
		break;
	}
}

void ST_SPI_send(ST_SPI_reg_t* pSpiReg, const uint8_t* data, const size_t data_len)
{
#define WAIT_FOR_TX while (!(pSpiReg->SR & 0x2))

	if ((data_len > 0) && (data != NULL))
	{
		const uint8_t dff = pSpiReg->CR1 & (1 << 11);
		if (dff == ST_SPI_DFF_8Bit)
		{
			for (size_t i = 0; i < data_len; ++i)
			{
				// wait for TX buffer to be available (empty)
				WAIT_FOR_TX;
				pSpiReg->DR = data[i];
			}
		}
		else
		{
			for (size_t i = 0; i < data_len; i += 2)
			{
				// wait for TX buffer to be available (empty)
				WAIT_FOR_TX;

				if (i < data_len - 1)
				{
					const uint16_t to_transfer = (data[i + 1] | (data[i] << 8));
					pSpiReg->DR = to_transfer;
				}
				else
				{
					pSpiReg->DR = data[i];
				}
			}
		}
	}

#undef WAIT_FOR_TX
}

void ST_SPI_recv(ST_SPI_reg_t* pSpiReg, uint8_t* data, const size_t data_len)
{
#define WAIT_FOR_RX while (!(pSpiReg->SR & 0x1))
	if ((data_len > 0) && (data != NULL))
	{
		const uint8_t dff = pSpiReg->CR1 & (1 << 11);
		if (dff == ST_SPI_DFF_8Bit)
		{
			for (size_t i = 0; i < data_len; ++i)
			{
				// wait for RX buffer to be available (not empty)
				WAIT_FOR_RX;
				data[i] = (uint8_t)pSpiReg->DR;
			}
		}
		else
		{
			for (size_t i = 0; i < data_len; i += 2)
			{
				// wait for RX buffer to be available (not empty)
				WAIT_FOR_RX;

				if (i < data_len - 1)
				{
					*((uint16_t*)(data + i)) = (uint16_t)pSpiReg->DR;
				}
				else
				{
					data[i] = (uint8_t)pSpiReg->DR;
				}
			}
		}
	}
#undef WAIT_FOR_RX
}

static int8_t ST_SPI_IRQ_get_numer(ST_SPI_reg_t * pSpiReg)
{
	switch ((uint32_t)pSpiReg)
	{
	case ST_SPI1_BASE_ADDRESS:
		return ST_NVIC_IRQ_SPI1;
	case ST_SPI2_BASE_ADDRESS:
		return ST_NVIC_IRQ_SPI2;
	case ST_SPI3_BASE_ADDRESS:
		return ST_NVIC_IRQ_SPI3;
	case ST_SPI4_BASE_ADDRESS:
		return ST_NVIC_IRQ_SPI4;
	case ST_SPI5_BASE_ADDRESS:
		return ST_NVIC_IRQ_SPI5;
	default:
		break;
	}
	return -1;
}

void ST_SPI_IRQ_control(ST_SPI_reg_t * pSpiReg, uint8_t priority, uint8_t enable)
{
	const int8_t irq_no = ST_SPI_IRQ_get_numer(pSpiReg);
	if (irq_no == -1)
	{
		return;
	}

	ST_NVIC_configure_interrupt(irq_no, priority, enable);
}

uint8_t ST_SPI_send_irq(ST_SPI_t *pSpi, const uint8_t* data, const size_t data_len)
{
	if (pSpi->irq.tx_state == ST_SPI_IRQ_IDLE)
	{
		pSpi->irq.tx_state = ST_SPI_IRQ_BUSY_TX;
		pSpi->irq.tx_buff = data;
		pSpi->irq.tx_buff_len = (uint32_t)data_len;
		pSpi->baseAddress->CR2 |= (1 << 7); // TXEIE bit
		pSpi->baseAddress->CR2 |= (1 << 5); // ERRIE bit
		return 1;
	}
	return 0;
}

uint8_t ST_SPI_recv_irq(ST_SPI_t *pSpi, uint8_t* data, const size_t data_len)
{
	if (pSpi->irq.rx_state == ST_SPI_IRQ_IDLE)
	{
		pSpi->irq.rx_state = ST_SPI_IRQ_BUSY_RX;
		pSpi->irq.rx_buff = data;
		pSpi->irq.rx_buff_len = (uint32_t)data_len;
		pSpi->baseAddress->CR2 |= (1 << 6); // RXEIE bit
		pSpi->baseAddress->CR2 |= (1 << 5); // ERRIE bit
		return 1;
	}
	return 0;
}

//TODO handle 16 bit DFF
void ST_SPI_IRQ_handle(ST_SPI_t *pSpi)
{
	if ((pSpi->baseAddress->SR & 0x1) && (pSpi->baseAddress->CR2 & (1 << 6)))
	{
		// Receive buffer not-empty
		if ((pSpi->irq.rx_state == ST_SPI_IRQ_BUSY_RX) && (pSpi->irq.rx_buff_len > 0))
		{
			*pSpi->irq.rx_buff = pSpi->baseAddress->DR;
			++pSpi->irq.rx_buff;
			--pSpi->irq.rx_buff_len;
			if (pSpi->irq.rx_buff_len == 0)
			{
				pSpi->irq.rx_buff = 0;
				pSpi->baseAddress->CR2 &= ~(1 << 6);
				pSpi->baseAddress->CR2 &= ~(1 << 5);
				pSpi->irq.rx_state = ST_SPI_IRQ_IDLE;
				if (ST_SPI_App_Event)
				{
					ST_SPI_App_Event(pSpi, ST_SPI_EVENT_RX_COMPL);
				}
			}
		}
	}

	if ((pSpi->baseAddress->SR & 0x2) && (pSpi->baseAddress->CR2 & (1 << 7)))
	{
		// transmit buffer ready
		if ((pSpi->irq.tx_state == ST_SPI_IRQ_BUSY_TX) && (pSpi->irq.tx_buff_len > 0))
		{
			pSpi->baseAddress->DR = *pSpi->irq.tx_buff;
			++pSpi->irq.tx_buff;
			--pSpi->irq.tx_buff_len;
			if (pSpi->irq.tx_buff_len == 0)
			{
				pSpi->irq.tx_buff = 0;
				pSpi->baseAddress->CR2 &= ~(1 << 7);
				pSpi->baseAddress->CR2 &= ~(1 << 5);
				pSpi->irq.tx_state = ST_SPI_IRQ_IDLE;
				if (ST_SPI_App_Event)
				{
					ST_SPI_App_Event(pSpi, ST_SPI_EVENT_TX_COMPL);
				}
			}
		}
	}
}

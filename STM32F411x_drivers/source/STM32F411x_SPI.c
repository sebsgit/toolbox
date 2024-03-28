#include "STM32F411x_SPI.h"

#define SET_BIT(where, which, value) do { if (value) { where |= (1 << (which)); } else { where &= ~(1 << (which)); } } while (0);

void ST_SPI_init(ST_SPI_t * pSpi)
{
	ST_SPI_clock_control(pSpi->baseAddress, 1);

	pSpi->baseAddress->CR1 &= ~(1 << 6); // disable SPI before applying new config
	pSpi->baseAddress->CR2 |= (1 << 4);		// SPI TI mode

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
#define WAIT_FOR_TX while ((pSpiReg->SR & 0x2) != 0x2)

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

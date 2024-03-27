#include "STM32F411x_SPI.h"

void ST_SPI_init(ST_SPI_t * pSpi)
{

}

void ST_SPI_deinit(ST_SPI_reg_t * pSpiReg)
{
	switch ((uint32_t)spiReg)
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

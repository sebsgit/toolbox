#include "STM32F411x_I2C.h"

void ST_I2C_init(ST_I2C_t *i2c)
{
	i2c->baseAdress->CR1 &= ~(1 << 0); // peripheral disable

	// configure slave address
	{
		i2c->baseAdress->OAR1 &= ~(1 << 15); // 7-bit slave addressing
		i2c->baseAdress->OAR1 |= (1 << 14); // bit 14 in OAR1 register should be kept at 1
		i2c->baseAdress->OAR1 &= ~(0x3F << 1); // clear slave address
		i2c->baseAdress->OAR1 |= (i2c->config.slave_address << 1); // set slave address
	}

	// configure ACK
	if (i2c->config.ack_enable)
	{
		i2c->baseAdress->CR1 |= (1 << 10);
	}
	else
	{
		i2c->baseAdress->CR1 &= ~(1 << 10);
	}

	// configure speed
	{
		if (i2c->config.mode == ST_I2C_MODE_STD)
		{
			i2c->baseAdress->CCR &= ~(1 << 15);
		}
		else
		{
			i2c->baseAdress->CCR |= (1 << 15);
			if (i2c->config.fm_duty_cycle == ST_I2C_FM_DUTY_CYCLE_2)
			{
				i2c->baseAdress->CCR &= ~(1 << 14);
			}
			else
			{
				i2c->baseAdress->CCR |= (1 << 14);
			}
		}
		//TODO configure CCR bits 0-10 according to the selected clock speed
	}

	i2c->baseAdress->CR1 |= (1 << 0); // peripheral enable
}

void ST_I2C_deinit(ST_I2C_t *i2c)
{
	i2c->baseAdress->CR1 |= (1 << 15); // peripheral software reset
}

#include "STM32F411x_I2C.h"
#include "STM32F411x_RCC.h"

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
		// configure clock frequency, bits 0-5 in CR2
		uint32_t sys_clock_freq = ST_RCC_get_system_clock_frequency();
		sys_clock_freq /= ST_RCC_get_ahb1_prescaler();
		sys_clock_freq /= ST_RCC_get_apb1_prescaler();
		i2c->baseAdress->CR2 &= ~(0x1F);
		i2c->baseAdress->CR2 |= ((sys_clock_freq / (1000U * 1000U)) & 0x1F);

		uint32_t ccr = 0;
		if (i2c->config.mode == ST_I2C_MODE_STD)
		{
			i2c->baseAdress->CCR &= ~(1 << 15);

			// T_low + T_high = 2 * (CCR * T_sys_clock) => [freq domain] => CCR = freq_sys_clock / (2 * freq_i2c)
			ccr = (sys_clock_freq) / (2U * i2c->config.mode);
		}
		else
		{
			i2c->baseAdress->CCR |= (1 << 15);
			if (i2c->config.fm_duty_cycle == ST_I2C_FM_DUTY_CYCLE_2)
			{
				i2c->baseAdress->CCR &= ~(1 << 14);
				ccr = (sys_clock_freq) / (3U * i2c->config.mode);
			}
			else
			{
				i2c->baseAdress->CCR |= (1 << 14);
				ccr = (sys_clock_freq) / (25U * i2c->config.mode);
			}
		}
		// Bits 11:0 CCR[11:0]: Clock control register in Fm/Sm mode (Master mode)
		i2c->baseAdress->CCR &= ~(0x7FF);
		i2c->baseAdress->CCR |= ccr & 0x7FF;
	}

	i2c->baseAdress->CR1 |= (1 << 0); // peripheral enable
}

void ST_I2C_deinit(ST_I2C_t *i2c)
{
	i2c->baseAdress->CR1 |= (1 << 15); // peripheral software reset
}

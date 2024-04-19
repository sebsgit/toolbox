#include "STM32F411x_RCC.h"

#include <stdint.h>

void ST_RCC_set_system_clock(ST_RCC_system_clock_t clock_source)
{
	uint32_t tmp = 0;
	tmp |= (uint32_t)clock_source;
	ST_RCC->CFG &= ~(0x3);
	ST_RCC->CFG |= tmp;
}

uint8_t ST_RCC_system_clock_select_status()
{
	uint8_t result = 0;
	result |= (ST_RCC->CFG >> 2) & 0x3;
	return result;
}

uint32_t ST_RCC_get_system_clock_frequency()
{
	const uint8_t active_clock = ST_RCC_system_clock_select_status();
	if (active_clock == ST_RCC_SYS_CLOCK_HSI)
	{
		return 16 * 1000 * 1000U;
	}
	else if (active_clock == ST_RCC_SYS_CLOCK_HSE)
	{
		return 8 * 1000 * 1000U;
	}
	else
	{
		//TODO PLL
		return 0U;
	}
}

uint32_t ST_RCC_get_apb1_prescaler()
{
	// Bits 12:10 PPRE1: APB Low speed prescaler (APB1)
	const uint32_t value = (ST_RCC->CFG >> 10) & 0x7;
	if (value < 4U)
	{
		return 1U;
	}
	else
	{
		return 1U << (1U + (value - 4U));
	}
}

uint32_t ST_RCC_get_ahb_prescaler()
{
	// Bits 7:4 HPRE: AHB prescaler
	const uint32_t value = (ST_RCC->CFG >> 4) & 0xF;
	if (value < 8U)
	{
		return 1U;
	}
	else
	{
		return 1U << (1U + (value - 8U));
	}
}

uint32_t ST_RCC_get_apb2_prescaler()
{
	// Bits 15:13 PPRE2: APB high-speed prescaler (APB2)
	const uint32_t value = (ST_RCC->CFG >> 13) & 0x7;
	if (value < 4U)
	{
		return 1U;
	}
	else
	{
		return 1U << (1U + (value - 4U));
	}
}

uint32_t ST_RCC_get_apb1_clock_frequency()
{
	uint32_t clock_freq = ST_RCC_get_system_clock_frequency();
	clock_freq /= ST_RCC_get_ahb_prescaler();
	clock_freq /= ST_RCC_get_apb1_prescaler();
	return clock_freq;
}

uint32_t ST_RCC_get_apb2_clock_frequency()
{
	uint32_t clock_freq = ST_RCC_get_system_clock_frequency();
	clock_freq /= ST_RCC_get_ahb_prescaler();
	clock_freq /= ST_RCC_get_apb2_prescaler();
	return clock_freq;
}

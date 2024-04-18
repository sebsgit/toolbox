#include "STM32F411x_GPIO.h"

#define SET_BIT(where, which, value) do { if (value) { where |= (1 << (which)); } else { where &= ~(1 << (which)); } } while (0);

void ST_NVIC_configure_interrupt(uint32_t irq_no, uint8_t priority, uint8_t enable)
{
	volatile uint32_t *prio_reg = ST_NVIC_GET_PRIO_REGISTER(irq_no);
	const uint32_t prio_bit_pos = (irq_no % 4);
	const uint32_t clr_mask = ~(0xFF << (prio_bit_pos * 8));
	const uint32_t new_prio_value = ((uint32_t)(priority << 4) << (prio_bit_pos * 8));
	uint32_t value = *prio_reg;
	value &= clr_mask;
	*prio_reg = (value | new_prio_value);

	uint32_t *nvic_seten;
	uint32_t *nvic_clren;
	if (irq_no < 32)
	{
		nvic_seten = (uint32_t*)ST_NVIC_SET_ENABLE_0;
		nvic_clren = (uint32_t*)ST_NVIC_CLR_ENABLE_0;
	}
	else if (irq_no < (32 * 2))
	{
		nvic_seten = (uint32_t*)ST_NVIC_SET_ENABLE_1;
		nvic_clren = (uint32_t*)ST_NVIC_CLR_ENABLE_1;
	}
	else if (irq_no < (32 * 3))
	{
		nvic_seten = (uint32_t*)ST_NVIC_SET_ENABLE_2;
		nvic_clren = (uint32_t*)ST_NVIC_CLR_ENABLE_2;
	}
	else if (irq_no < (32 * 4))
	{
		nvic_seten = (uint32_t*)ST_NVIC_SET_ENABLE_3;
		nvic_clren = (uint32_t*)ST_NVIC_CLR_ENABLE_3;
	}
	//TODO

	if (enable)
	{
		*nvic_seten |= (1 << (irq_no % 32));
	}
	else
	{
		*nvic_clren |= (1 << (irq_no % 32));
	}
}

static uint32_t get_exti_bit_pattern(ST_GPIO_reg_t * pGpioReg)
{
	switch ((uint32_t)pGpioReg)
	{
	case ST_GPIO_A_BASE_ADDRESS:
		return ST_EXTI_CR_SEL_PORT_A;
	case ST_GPIO_B_BASE_ADDRESS:
		return ST_EXTI_CR_SEL_PORT_B;
	case ST_GPIO_C_BASE_ADDRESS:
		return ST_EXTI_CR_SEL_PORT_C;
	case ST_GPIO_D_BASE_ADDRESS:
		return ST_EXTI_CR_SEL_PORT_D;
	case ST_GPIO_E_BASE_ADDRESS:
		return ST_EXTI_CR_SEL_PORT_E;
	case ST_GPIO_H_BASE_ADDRESS:
		return ST_EXTI_CR_SEL_PORT_H;
	default:
		break;
	}

	return 0;
}

void ST_GPIO_init(ST_GPIO_t * pGpio)
{
	ST_GPIO_clock_control(pGpio->portAddr, 1);

	SET_BIT(pGpio->portAddr->PUPD, 2 * pGpio->config.pin, pGpio->config.pull_up & 0x1);
	SET_BIT(pGpio->portAddr->PUPD, 2 * pGpio->config.pin + 1, pGpio->config.pull_up & 0x2);

	if (pGpio->config.mode < ST_GPIO_PIN_MODE_ALTERNATE)
	{
		SET_BIT(pGpio->portAddr->MODE, 2 * pGpio->config.pin, pGpio->config.mode & 0x1);
		SET_BIT(pGpio->portAddr->MODE, 2 * pGpio->config.pin + 1, pGpio->config.mode & 0x2);
		if (pGpio->config.mode == ST_GPIO_PIN_MODE_OUTPUT)
		{
			SET_BIT(pGpio->portAddr->OTYPE, pGpio->config.pin, pGpio->config.output_type);
			SET_BIT(pGpio->portAddr->OSPEED, 2 * pGpio->config.pin, pGpio->config.output_speed & 0x1);
			SET_BIT(pGpio->portAddr->OSPEED, 2 * pGpio->config.pin + 1, pGpio->config.output_speed & 0x2);
		}
	}
	else if (pGpio->config.mode == ST_GPIO_PIN_MODE_ALTERNATE)
	{
		SET_BIT(pGpio->portAddr->MODE, 2 * pGpio->config.pin, pGpio->config.mode & 0x1);
		SET_BIT(pGpio->portAddr->MODE, 2 * pGpio->config.pin + 1, pGpio->config.mode & 0x2);
		if (pGpio->config.pin > 7)
		{
			SET_BIT(pGpio->portAddr->AFRH, 4 * (pGpio->config.pin - 8), pGpio->config.alternate_function & 0x1);
			SET_BIT(pGpio->portAddr->AFRH, 4 * (pGpio->config.pin - 8) + 1, pGpio->config.alternate_function & 0x2);
			SET_BIT(pGpio->portAddr->AFRH, 4 * (pGpio->config.pin - 8) + 2, pGpio->config.alternate_function & 0x4);
			SET_BIT(pGpio->portAddr->AFRH, 4 * (pGpio->config.pin - 8) + 3, pGpio->config.alternate_function & 0x8);
		}
		else
		{
			SET_BIT(pGpio->portAddr->AFRL, 4 * pGpio->config.pin, pGpio->config.alternate_function & 0x1);
			SET_BIT(pGpio->portAddr->AFRL, 4 * pGpio->config.pin + 1, pGpio->config.alternate_function & 0x2);
			SET_BIT(pGpio->portAddr->AFRL, 4 * pGpio->config.pin + 2, pGpio->config.alternate_function & 0x4);
			SET_BIT(pGpio->portAddr->AFRL, 4 * pGpio->config.pin + 3, pGpio->config.alternate_function & 0x8);
		}
	}
	else if (pGpio->config.mode > ST_GPIO_PIN_MODE_ANALOG)
	{
		//
		// configure interrupts
		//
		ST_SYSCFG_CLOCK_EN();

		if (pGpio->config.mode == ST_GPIO_PIN_MODE_IRQ_R)
		{
			ST_EXTI->RTSR |= (1 << pGpio->config.pin);
			ST_EXTI->FTSR &= ~(1 << pGpio->config.pin);
			ST_EXTI->IMR |= (1 << pGpio->config.pin);
		}
		else if (pGpio->config.mode == ST_GPIO_PIN_MODE_IRQ_F)
		{
			ST_EXTI->RTSR &= ~(1 << pGpio->config.pin);
			ST_EXTI->FTSR |= (1 << pGpio->config.pin);
			ST_EXTI->IMR |= (1 << pGpio->config.pin);
		}
		else if (pGpio->config.mode == ST_GPIO_PIN_MODE_IRQ_RF)
		{
			ST_EXTI->FTSR |= (1 << pGpio->config.pin);
			ST_EXTI->RTSR |= (1 << pGpio->config.pin);
			ST_EXTI->IMR |= (1 << pGpio->config.pin);
		}

		const uint8_t exti_line = ST_EXTI_GET_CRIDX(pGpio->config.pin);
		const uint8_t exti_bit_pos = pGpio->config.pin % 4;
		const uint32_t exti_bits = get_exti_bit_pattern(pGpio->portAddr) << (4 * exti_bit_pos);
		uint32_t value = ST_SYSCFG->EXTICONF[exti_line];
		value &= ~( 0xF << (4 * exti_bit_pos) );
		ST_SYSCFG->EXTICONF[exti_line] = value | exti_bits;
	}
}

void ST_GPIO_deinit(ST_GPIO_reg_t * pGpioReg)
{
	switch ((uint32_t)pGpioReg)
	{
	case ST_GPIO_A_BASE_ADDRESS:
		ST_RCC->AHB1RST |= (1 << 0);
		ST_RCC->AHB1RST &= ~(1 << 0);
		break;
	case ST_GPIO_B_BASE_ADDRESS:
		ST_RCC->AHB1RST |= (1 << 1);
		ST_RCC->AHB1RST &= ~(1 << 1);
		break;
	case ST_GPIO_C_BASE_ADDRESS:
		ST_RCC->AHB1RST |= (1 << 2);
		ST_RCC->AHB1RST &= ~(1 << 2);
		break;
	case ST_GPIO_D_BASE_ADDRESS:
		ST_RCC->AHB1RST |= (1 << 3);
		ST_RCC->AHB1RST &= ~(1 << 3);
		break;
	case ST_GPIO_E_BASE_ADDRESS:
		ST_RCC->AHB1RST |= (1 << 4);
		ST_RCC->AHB1RST &= ~(1 << 4);
		break;
	case ST_GPIO_H_BASE_ADDRESS:
		ST_RCC->AHB1RST |= (1 << 7);
		ST_RCC->AHB1RST &= ~(1 << 7);
		break;
	default:
		break;
	}
}

void ST_GPIO_clock_control(ST_GPIO_reg_t *pGpioReg, uint8_t enable)
{
	switch ((uint32_t)pGpioReg)
		{
		case ST_GPIO_A_BASE_ADDRESS:
			if (enable)
			{
				ST_GPIOA_CLCK_EN();
			}
			else
			{
				ST_GPIOA_CLCK_DI();
			}
			break;
		case ST_GPIO_B_BASE_ADDRESS:
			if (enable)
			{
				ST_GPIOB_CLCK_EN();
			}
			else
			{
				ST_GPIOB_CLCK_DI();
			}
			break;
		case ST_GPIO_C_BASE_ADDRESS:
			if (enable)
			{
				ST_GPIOC_CLCK_EN();
			}
			else
			{
				ST_GPIOC_CLCK_DI();
			}
			break;
		case ST_GPIO_D_BASE_ADDRESS:
			if (enable)
			{
				ST_GPIOD_CLCK_EN();
			}
			else
			{
				ST_GPIOD_CLCK_DI();
			}
			break;
		case ST_GPIO_E_BASE_ADDRESS:
			if (enable)
			{
				ST_GPIOE_CLCK_EN();
			}
			else
			{
				ST_GPIOE_CLCK_DI();
			}
			break;
		case ST_GPIO_H_BASE_ADDRESS:
			if (enable)
			{
				ST_GPIOH_CLCK_EN();
			}
			else
			{
				ST_GPIOH_CLCK_DI();
			}
			break;
		default:
			break;
		}
}

uint8_t ST_GPIO_read_pin(ST_GPIO_reg_t *pGpioReg, uint8_t pin)
{
	return (pGpioReg->ID & (1 << pin)) ? 1 : 0;
}

uint16_t ST_GPIO_read_port(ST_GPIO_reg_t *pGpioReg)
{
	return pGpioReg->ID & 0x0000FFFF;
}

void ST_GPIO_write_pin(ST_GPIO_reg_t *pGpio, uint8_t pin, uint8_t value)
{
	if (value)
	{
		pGpio->OD |= (1 << pin);
	}
	else
	{
		pGpio->OD &= ~(1 << pin);
	}
}

void ST_GPIO_write_port(ST_GPIO_reg_t *pGpio, uint16_t value)
{
	pGpio->OD = value;
}

int32_t ST_GPIO_IRQ_get_numer(uint8_t pin)
{
	if (pin == 0)
	{
		return ST_NVIC_IRQ_EXTI_0;
	}
	else if (pin == 1)
	{
		return ST_NVIC_IRQ_EXTI_1;
	}
	else if (pin == 2)
	{
		return ST_NVIC_IRQ_EXTI_2;
	}
	else if (pin == 3)
	{
		return ST_NVIC_IRQ_EXTI_3;
	}
	else if (pin == 4)
	{
		return ST_NVIC_IRQ_EXTI_4;
	}
	else if (pin >= 5 && pin <= 9)
	{
		return ST_NVIC_IRQ_EXTI9_5;
	}
	else if (pin >= 10 && pin <= 15)
	{
		return ST_NVIC_IRQ_EXTI15_10;
	}

	return -1;
}

void ST_GPIO_IRQ_control(uint8_t pin, uint8_t priority, uint8_t enable)
{
	const int8_t irq_no = ST_GPIO_IRQ_get_numer(pin);
	if (irq_no == -1)
	{
		return;
	}

	ST_NVIC_configure_interrupt(irq_no, priority, enable);
}

void ST_GPIO_IRQ_handle(uint8_t pin)
{
	if (ST_EXTI->PR & (1 << pin))
	{
		ST_EXTI->PR |= (1 << pin);
	}
}


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

		// configure T-rise register
		uint32_t t_rise;
		if (i2c->config.mode == ST_I2C_MODE_STD)
		{
			t_rise = (sys_clock_freq / (1000U * 1000U)) + 1;
		}
		else
		{
			t_rise = ((sys_clock_freq * 300U) / (1000U * 1000U * 1000U)) + 1;
		}
		i2c->baseAdress->TRISE = t_rise & 0x3F;
	}

	i2c->baseAdress->CR1 |= (1 << 0); // peripheral enable

	// configure ACK
	if (i2c->config.ack_enable)
	{
		i2c->baseAdress->CR1 |= (1 << 10);
	}
	else
	{
		i2c->baseAdress->CR1 &= ~(1 << 10);
	}
}

void ST_I2C_deinit(ST_I2C_t *i2c)
{
	i2c->baseAdress->CR1 |= (1 << 15); // peripheral software reset
}

void ST_I2C_Master_send(ST_I2C_t *i2c, const uint8_t slave_addr, const uint8_t *tx_buffer, const size_t data_len)
{
	// clear ACK failure
	i2c->baseAdress->SR1 &= ~(1 << 10);

	// generate start condition
	i2c->baseAdress->CR1 |= (1 << 8);

	// wait for start condition confirmation
	// TODO: add timeout
	while (!(i2c->baseAdress->SR1 & 0x1));

	// set RW bit to "W = 0" and send the slave address
	uint8_t slave_addr_shifted = (slave_addr << 1);
	slave_addr_shifted &= ~(1);
	i2c->baseAdress->DR = slave_addr_shifted;

	// wait for address sent confirmation
	while (!(i2c->baseAdress->SR1 & 0x2))
	{
		if (i2c->baseAdress->SR1 & (1 << 10)) // ACK failure
		{
			//TODO add return codes
			return;
		}
	}

	// clear ADDR flag by reading SR1 and SR2
	uint32_t unused = i2c->baseAdress->SR1;
	unused = i2c->baseAdress->SR2;
	(void)unused;

	// send data
	//TODO: error checking
	for (size_t i = 0; i < data_len; ++i)
	{
		// wait for TxE
		while (!(i2c->baseAdress->SR1 & (1 << 7)));

		// write DR
		i2c->baseAdress->DR = tx_buffer[i];
	}

	// data transmission finalization: wait for TxE=1 and BTF=1
	while (!(i2c->baseAdress->SR1 & (1 << 7)));
	while (!(i2c->baseAdress->SR1 & (1 << 2)));

	// generate stop condition
	i2c->baseAdress->CR1 |= (1 << 9);
}

void ST_I2C_Master_receive(ST_I2C_t *i2c, const uint8_t slave_addr, uint8_t *rx_buffer, const size_t data_len)
{
	//TODO implement according to the transmission state diagram,
	// Reference manual "STM32F411xC/E advanced Arm-based 32-bit MCUs", page 482
}

void ST_I2C_irq_control(ST_I2C_t *i2c, uint8_t priority, uint8_t enable)
{

}

uint8_t ST_I2C_Master_send_IT(ST_I2C_t *i2c, const uint8_t slave_addr, const uint8_t *tx_buffer, const size_t data_len)
{
	const uint8_t busy = i2c->irq.irq_state;
	if (busy == ST_I2C_IRQ_STATE_IDLE)
	{
		i2c->irq.dev_addr = slave_addr;
		i2c->irq.irq_state = ST_I2C_IRQ_STATE_BUSY_TX;
		i2c->irq.tx_buffer = tx_buffer;
		i2c->irq.tx_buff_len = data_len;

		// enable event, error and buffer interrupts
		i2c->baseAdress->CR2 |= (1 << 10);
		i2c->baseAdress->CR2 |= (1 << 9);
		i2c->baseAdress->CR2 |= (1 << 8);

		// generate start condition
		i2c->baseAdress->CR1 |= (1 << 8);

		return ST_I2C_IRQ_STATE_BUSY_TX;
	}
	return busy;
}

uint8_t ST_I2C_Master_receive_IT(ST_I2C_t *i2c, const uint8_t slave_addr, uint8_t *rx_buffer, const size_t data_len)
{
	const uint8_t busy = i2c->irq.irq_state;
	if (busy == ST_I2C_IRQ_STATE_IDLE)
	{
		i2c->irq.dev_addr = slave_addr;
		i2c->irq.irq_state = ST_I2C_IRQ_STATE_BUSY_RX;
		i2c->irq.rx_buffer = rx_buffer;
		i2c->irq.rx_buff_len = data_len;
		i2c->irq.rx_size = data_len;

		// enable event, error and buffer interrupts
		i2c->baseAdress->CR2 |= (1 << 10);
		i2c->baseAdress->CR2 |= (1 << 9);
		i2c->baseAdress->CR2 |= (1 << 8);

		// generate start condition
		i2c->baseAdress->CR1 |= (1 << 8);

		return ST_I2C_IRQ_STATE_BUSY_RX;
	}
	return busy;
}



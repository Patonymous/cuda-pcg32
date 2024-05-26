#pragma once

#include <iostream>

class indent
{
public:
	unsigned level;
	std::string indentation;

	inline indent(const std::string& _indentation = "\t") : level(0), indentation(_indentation) { }

	inline indent& operator++()
	{
		level++;
		return *this;
	}
	inline indent operator++(int)
	{
		indent copy = *this;
		level++;
		return copy;
	}

	inline indent& operator--()
	{
		level--;
		return *this;
	}
	inline indent operator--(int)
	{
		indent copy = *this;
		level--;
		return copy;
	}
};

inline std::ostream& operator<<(std::ostream& os, const indent& ind)
{
	for (unsigned i = 0; i < ind.level; i++)
		os << ind.indentation;
	return os;
}
